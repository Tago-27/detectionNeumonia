"""
API de detección de neumonía + recomendación generada con LLM + aprendizaje por refuerzo con el feedback del usuario.

Pipeline:  imagen --POST /predecir--> preprocesamiento --> CNN (predicción)
           --> agente RL elige el estilo --> LLM genera la recomendación con la predicción como input --> JSON
           usuario: me gusta / no me gusta --POST /rl/feedback--> agente RL actualiza su tabla Q

Ejecutar (desde api/):  uvicorn main:app --reload
Variables de entorno: GEMINI_API_KEY (LLM) y DATABASE_URL (tabla Q en Postgres). Sin ellas funciona
igual, con textos de respaldo y la tabla Q en datos/q_table_recomendaciones.json.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agente_rl import AgenteRecomendaciones, AlmacenArchivo, AlmacenPostgres
from preprocesamiento import bytes_a_tensor, construir_filtro
from recomendaciones_llm import cargar_llm, generar_recomendacion, llm_cargado

BASE = Path(__file__).resolve().parent.parent          # carpeta raíz del proyecto
CARPETA_MODELO = BASE / "modelo"
RUTA_WEB = BASE / "web" / "index.html"
RUTA_Q_TABLE = BASE / "datos" / "q_table_recomendaciones.json"


def _unico(patron: str) -> Path:
    """Busca el archivo del modelo aprobado en modelo/, tenga el nombre que tenga."""
    encontrados = sorted(CARPETA_MODELO.glob(patron))
    if len(encontrados) != 1:
        raise FileNotFoundError(
            f"Se esperaba exactamente un archivo '{patron}' en {CARPETA_MODELO} y hay {len(encontrados)}: "
            f"{[p.name for p in encontrados]}. Deja solo el modelo aprobado y su config."
        )
    return encontrados[0]


class ModeloTFLite:
    """CNN convertida a .tflite (herramientas/convertir_a_tflite.py) y ejecutada con LiteRT.

    Da las mismas probabilidades que el .keras usando mucha menos RAM que TensorFlow completo,
    lo que permite desplegar en Render gratis (512 MB).
    """

    def __init__(self, ruta: Path):
        self._interprete = Interpreter(model_path=str(ruta), num_threads=1)
        self._interprete.allocate_tensors()
        self._entrada = self._interprete.get_input_details()[0]["index"]
        self._salida = self._interprete.get_output_details()[0]["index"]
        self._candado = threading.Lock()   # el intérprete no admite dos predicciones a la vez

    def predecir(self, x: np.ndarray) -> np.ndarray:
        with self._candado:
            self._interprete.set_tensor(self._entrada, x.astype(np.float32))
            self._interprete.invoke()
            return self._interprete.get_tensor(self._salida)[0].copy()


# 1) La configuración, el modelo, el LLM y el agente se cargan UNA sola vez, al arrancar el servidor.
RUTA_MODELO = _unico("*.tflite")
RUTA_CONFIG = _unico("config*.json")
CONFIG = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
MODELO = ModeloTFLite(RUTA_MODELO)
FILTRO = construir_filtro(CONFIG)
CLASES = CONFIG["clases"]
UMBRAL = float(CONFIG["umbral"])
TAMANO_MAXIMO = 10 * 1024 * 1024   # 10 MB

cargar_llm()
# En Render gratis los archivos se borran al apagar el servicio: con DATABASE_URL la tabla Q va a Postgres.
URL_BD = os.getenv("DATABASE_URL")
AGENTE = AgenteRecomendaciones(AlmacenPostgres(URL_BD) if URL_BD else AlmacenArchivo(RUTA_Q_TABLE))
print(f"Agente RL: tabla Q en {AGENTE.almacen.descripcion} ({AGENTE.total_feedback} feedback acumulados)")

app = FastAPI(title="API Neumonía CNN + LLM + RL", version="3.0")
# CORS permite que una página abierta desde otro origen (p. ej. otro puerto) llame a la API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Feedback(BaseModel):
    estado: str
    accion: str
    recompensa: float   # +1 me gusta, -1 no me gusta (o un valor intermedio)


@app.get("/")
def pagina():
    """Entrega la interfaz web (HTML/CSS/JS)."""
    return FileResponse(RUTA_WEB)


@app.get("/salud")
def salud():
    """Permite comprobar que el servidor, el modelo y el LLM están listos."""
    return {"estado": "ok", "modelo": RUTA_MODELO.name, "filtro": CONFIG["filtro"], "umbral": UMBRAL,
            "llm_cargado": llm_cargado(), "tabla_q": AGENTE.almacen.descripcion,
            "feedback_recibido": AGENTE.total_feedback}


# def (no async): la predicción y el LLM bloquean la CPU, así FastAPI los ejecuta en otro hilo.
@app.post("/predecir")
def predecir(imagen: UploadFile = File(...)):
    contenido = imagen.file.read(TAMANO_MAXIMO + 1)
    if not contenido:
        raise HTTPException(status_code=400, detail="No se recibió ninguna imagen.")
    if len(contenido) > TAMANO_MAXIMO:
        raise HTTPException(status_code=413, detail="La imagen supera 10 MB.")
    try:
        x = bytes_a_tensor(contenido, CONFIG, FILTRO)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    prob = MODELO.predecir(x)                        # [P(NORMAL), P(PNEUMONIA)], ya es softmax
    prob_normal, prob_neumonia = float(prob[0]), float(prob[1])
    diagnostico = CLASES[1] if prob_neumonia >= UMBRAL else CLASES[0]

    # 2) El agente RL elige el estilo de la recomendación según el diagnóstico y la confianza.
    rl = AGENTE.elegir_estilo(AGENTE.estado(diagnostico, max(prob_normal, prob_neumonia)))
    # 3) La predicción entra como input al LLM, que genera la recomendación (> 200 caracteres).
    recomendacion = generar_recomendacion(diagnostico, prob_neumonia, rl["accion"])

    return {
        "diagnostico": diagnostico,
        "prob_normal": round(prob_normal, 4),
        "prob_neumonia": round(prob_neumonia, 4),
        "umbral": round(UMBRAL, 4),
        **recomendacion,
        "rl": rl,
    }


@app.post("/rl/feedback")
def feedback(f: Feedback):
    """El usuario califica la recomendación y el agente actualiza su tabla Q."""
    try:
        datos = AGENTE.registrar_feedback(f.estado, f.accion, f.recompensa)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:   # la base de datos no respondió: no se pierde nada, el usuario puede reintentar
        print(f"Error guardando la tabla Q: {err}")
        raise HTTPException(status_code=503, detail="No se pudo guardar la calificación. Inténtalo de nuevo.")
    return {"estado": "ok", "mensaje": "Gracias, su calificación actualizó al agente.", "datos": datos}


@app.get("/rl/estado")
def estado_rl():
    """Muestra lo que ha aprendido el agente (tabla Q, epsilon y feedback recibido)."""
    return AGENTE.resumen()
