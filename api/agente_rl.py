"""
Agente de aprendizaje por refuerzo (Q-learning) que aprende qué estilo de recomendación prefieren los usuarios.

- ESTADO: diagnóstico + nivel de confianza (por ejemplo "PNEUMONIA|alta").
- ACCIÓN: el estilo de la recomendación que genera el LLM.
- RECOMPENSA: el feedback del usuario (+1 me gusta, -1 no me gusta, o un valor intermedio).

La tabla Q se guarda para no perder lo aprendido al reiniciar la API:
- con DATABASE_URL (Render): en Postgres, porque Render gratis borra los archivos al apagar el servicio;
- sin DATABASE_URL (local): en un archivo JSON.
"""
import json
import random
import threading
from pathlib import Path
from typing import Dict, Optional

from recomendaciones_llm import ESTILOS

DIAGNOSTICOS = ["NORMAL", "PNEUMONIA"]
NIVELES = ["alta", "media", "baja"]


def nivel_confianza(prob_clase: float) -> str:
    if prob_clase >= 0.85:
        return "alta"
    if prob_clase >= 0.65:
        return "media"
    return "baja"


# ---------------------------------------------------------------- almacenamiento de la tabla Q
class AlmacenArchivo:
    """Guarda la tabla Q en un archivo JSON (desarrollo local)."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.descripcion = f"archivo {self.ruta}"

    def cargar(self) -> Optional[dict]:
        return json.loads(self.ruta.read_text(encoding="utf-8")) if self.ruta.exists() else None

    def guardar(self, datos: dict) -> None:
        # Se escribe en un temporal y luego se reemplaza: si el proceso muere a mitad, el JSON no se corrompe.
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        temporal = self.ruta.with_suffix(".tmp")
        temporal.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
        temporal.replace(self.ruta)


class AlmacenPostgres:
    """Guarda la tabla Q como una fila JSONB en Postgres (Neon, Supabase, etc.)."""

    CLAVE = "recomendaciones"

    def __init__(self, url: str):
        self.url = url
        self.descripcion = "Postgres"
        with self._conectar() as conexion:
            conexion.execute("CREATE TABLE IF NOT EXISTS agente_rl ("
                             " id TEXT PRIMARY KEY,"
                             " datos JSONB NOT NULL,"
                             " actualizado TIMESTAMPTZ NOT NULL DEFAULT now())")

    def _conectar(self):
        import psycopg
        # Una conexión por operación: el feedback es poco frecuente y Neon cierra las conexiones inactivas.
        return psycopg.connect(self.url, connect_timeout=15)

    def cargar(self) -> Optional[dict]:
        with self._conectar() as conexion:
            fila = conexion.execute("SELECT datos FROM agente_rl WHERE id = %s", (self.CLAVE,)).fetchone()
        return fila[0] if fila else None

    def guardar(self, datos: dict) -> None:
        from psycopg.types.json import Jsonb
        with self._conectar() as conexion:
            conexion.execute("INSERT INTO agente_rl (id, datos) VALUES (%s, %s) "
                             "ON CONFLICT (id) DO UPDATE SET datos = EXCLUDED.datos, actualizado = now()",
                             (self.CLAVE, Jsonb(datos)))


# ---------------------------------------------------------------- agente
class AgenteRecomendaciones:
    def __init__(self, almacen, alfa: float = 0.2, gamma: float = 0.0, epsilon: float = 0.2,
                 epsilon_min: float = 0.05, decaimiento: float = 0.99):
        self.almacen = almacen
        self.alfa, self.gamma = alfa, gamma
        self.epsilon, self.epsilon_min, self.decaimiento = epsilon, epsilon_min, decaimiento
        self.acciones = list(ESTILOS)
        self.total_feedback = 0
        self.q_table: Dict[str, Dict[str, float]] = {f"{d}|{n}": {a: 0.0 for a in self.acciones}
                                                     for d in DIAGNOSTICOS for n in NIVELES}
        self._candado = threading.Lock()   # FastAPI atiende peticiones en varios hilos
        # Si el almacén no responde, la API no arranca: así nunca se sobrescribe lo aprendido con una tabla vacía.
        datos = almacen.cargar()
        if datos:
            for estado, valores in datos.get("q_table", {}).items():
                self.q_table.setdefault(estado, {}).update({a: float(v) for a, v in valores.items() if a in self.acciones})
            self.total_feedback = int(datos.get("total_feedback", 0))
            self.epsilon = float(datos.get("epsilon", self.epsilon))

    def _datos(self) -> dict:
        return {"q_table": self.q_table, "total_feedback": self.total_feedback, "epsilon": self.epsilon}

    def estado(self, diagnostico: str, prob_clase: float) -> str:
        return f"{diagnostico}|{nivel_confianza(prob_clase)}"

    def elegir_estilo(self, estado: str) -> Dict[str, object]:
        """Política epsilon-greedy: a veces explora un estilo al azar, si no usa el de mayor valor Q."""
        with self._candado:
            if random.random() < self.epsilon:
                accion, modo = random.choice(self.acciones), "exploracion"
            else:
                valores = self.q_table[estado]
                mejor = max(valores.values())
                accion, modo = random.choice([a for a, v in valores.items() if v == mejor]), "explotacion"
            return {"estado": estado, "accion": accion, "modo": modo, "epsilon": round(self.epsilon, 4),
                    "valores_q": dict(self.q_table[estado])}

    def registrar_feedback(self, estado: str, accion: str, recompensa: float) -> Dict[str, object]:
        if estado not in self.q_table or accion not in self.acciones:
            raise ValueError("Estado o acción no válidos.")
        recompensa = max(-1.0, min(1.0, float(recompensa)))
        with self._candado:
            anterior, epsilon_anterior = self.q_table[estado][accion], self.epsilon
            objetivo = recompensa + self.gamma * max(self.q_table[estado].values())   # con gamma = 0: solo la recompensa
            self.q_table[estado][accion] = anterior + self.alfa * (objetivo - anterior)   # ecuación de Q-learning
            self.total_feedback += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decaimiento)
            try:
                self.almacen.guardar(self._datos())
            except Exception:
                # Si no se pudo guardar, se deshace en memoria para que memoria y almacén no diverjan.
                self.q_table[estado][accion], self.epsilon = anterior, epsilon_anterior
                self.total_feedback -= 1
                raise
            return {"estado": estado, "accion": accion, "recompensa": recompensa,
                    "q_anterior": round(anterior, 4), "q_nuevo": round(self.q_table[estado][accion], 4),
                    "total_feedback": self.total_feedback, "epsilon": round(self.epsilon, 4)}

    def resumen(self) -> Dict[str, object]:
        with self._candado:
            return {"total_feedback": self.total_feedback, "epsilon": round(self.epsilon, 4), "alfa": self.alfa,
                    "almacenamiento": self.almacen.descripcion,
                    "q_table": {e: dict(v) for e, v in self.q_table.items()}}
