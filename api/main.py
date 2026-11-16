"""
API de detección de neumonía + recomendación generada con LLM + aprendizaje por refuerzo con el feedback del usuario.

Pipeline:  imagen --POST /predecir--> filtro CLIP (¿es radiografía? si no, 422) --> preprocesamiento
           --> CNN (predicción + Grad-CAM)
           --> agente RL elige el estilo --> LLM genera la recomendación con la predicción como input --> JSON
           usuario: 1 a 5 estrellas --POST /rl/feedback--> agente RL actualiza su tabla Q
           GET /rl/historial --> recompensas y promedio móvil (gráfica de la web)
           POST /asistente: diagnóstico + probabilidad + pregunta --> LLM --> respuesta (máx. 4 oraciones)

Ejecutar (desde api/):  uvicorn main:app --reload
Variables de entorno: GEMINI_API_KEY (LLM) y DATABASE_URL (tabla Q en Postgres). Sin ellas funciona
igual, con textos de respaldo y la tabla Q en datos/q_table_recomendaciones.json.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

# Sin terminal (Render), Python acumula lo que imprime y los logs no se ven hasta el final o nunca:
# así cada línea aparece en los logs al momento.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
_INICIO = time.time()


def log_arranque(mensaje: str) -> None:
    print(f"[arranque {time.time() - _INICIO:5.1f}s] {mensaje}")


log_arranque("importando librerías...")
import numpy as np  # noqa: E402
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Literal, Optional

from pydantic import BaseModel, Field

import asistente
from agente_rl import AgenteRecomendaciones, AlmacenArchivo, AlmacenPostgres
import filtro_clip
import gradcam
from preprocesamiento import bytes_a_gris, construir_filtro, gris_a_tensor
import recomendaciones_llm
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
        salidas = self._interprete.get_output_details()
        # Salidas: probabilidades (1, 2) y, si el .tflite se generó para Grad-CAM, activaciones de relu6b (1, 7, 7, 512).
        self._salida = next(s["index"] for s in salidas if len(s["shape"]) == 2)
        self._activaciones = next((s["index"] for s in salidas if len(s["shape"]) == 4), None)
        self._candado = threading.Lock()   # el intérprete no admite dos predicciones a la vez

    def predecir(self, x: np.ndarray):
        """Devuelve (probabilidades, activaciones de la última capa convolucional o None)."""
        with self._candado:
            self._interprete.set_tensor(self._entrada, x.astype(np.float32))
            self._interprete.invoke()
            prob = self._interprete.get_tensor(self._salida)[0].copy()
            act = (self._interprete.get_tensor(self._activaciones)[0].copy()
                   if self._activaciones is not None else None)
            return prob, act


# 1) La configuración, el modelo, el LLM y el agente se cargan UNA sola vez, al arrancar el servidor.
RUTA_MODELO = _unico("*.tflite")
RUTA_CONFIG = _unico("config*.json")
CONFIG = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
log_arranque(f"cargando la CNN ({RUTA_MODELO.name})...")
MODELO = ModeloTFLite(RUTA_MODELO)
FILTRO = construir_filtro(CONFIG)
# Grad-CAM: pesos de la cabeza densa (modelo/*_gradcam.npz). Si faltan, la API funciona igual pero sin mapa de calor.
_npz = sorted(CARPETA_MODELO.glob("*_gradcam.npz"))
CABEZA_GRADCAM = gradcam.CabezaGradCAM(_npz[0]) if len(_npz) == 1 and MODELO._activaciones is not None else None
log_arranque("Grad-CAM " + ("listo" if CABEZA_GRADCAM else "NO disponible (falta el .npz o la salida relu6b del .tflite)"))
CLASES = CONFIG["clases"]
UMBRAL = float(CONFIG["umbral"])
TAMANO_MAXIMO = 10 * 1024 * 1024   # 10 MB
METRICAS = asistente.metricas_desde_config(CONFIG)   # confiabilidad del modelo, para el asistente

# Filtro CLIP "¿es una radiografía?". Si no carga (o USAR_CLIP=0), se dejan pasar todas las imágenes.
UMBRAL_CLIP = float(os.getenv("UMBRAL_CLIP", "0.5"))
FILTRO_CLIP = None
if os.getenv("USAR_CLIP", "1") == "1":
    log_arranque("cargando el filtro CLIP (ONNX int8)...")
    try:
        FILTRO_CLIP = filtro_clip.FiltroCLIP(CARPETA_MODELO / "clip_vision_int8.onnx",
                                             CARPETA_MODELO / "clip_textos.npz", UMBRAL_CLIP)
        log_arranque(f"filtro CLIP listo (umbral {UMBRAL_CLIP})")
    except Exception as err:
        log_arranque(f"filtro CLIP NO disponible, se aceptarán todas las imágenes: {type(err).__name__}: {err}")
else:
    log_arranque("filtro CLIP desactivado (USAR_CLIP=0)")

log_arranque("conectando con Gemini...")
cargar_llm()
# En Render gratis los archivos se borran al apagar el servicio: con DATABASE_URL la tabla Q va a Postgres.
URL_BD = os.getenv("DATABASE_URL")
log_arranque("cargando la tabla Q del agente RL" + (" desde Postgres..." if URL_BD else "..."))
AGENTE = AgenteRecomendaciones(AlmacenPostgres(URL_BD) if URL_BD else AlmacenArchivo(RUTA_Q_TABLE))
log_arranque(f"agente RL listo: tabla Q en {AGENTE.almacen.descripcion} ({AGENTE.total_feedback} feedback acumulados)")
log_arranque("API lista, abriendo el puerto")

app = FastAPI(title="API Neumonía CNN + LLM + RL", version="3.0")
# CORS permite que una página abierta desde otro origen (p. ej. otro puerto) llame a la API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class Feedback(BaseModel):
    estado: str
    accion: str
    estrellas: Optional[int] = Field(None, ge=1, le=5)   # 1 a 5 estrellas (lo que envía la web)
    recompensa: Optional[float] = None                   # alternativa directa: -1 a +1


class Pregunta(BaseModel):
    diagnostico: Literal["NORMAL", "PNEUMONIA"]
    prob_neumonia: float = Field(..., ge=0, le=1)
    pregunta: str = Field(..., min_length=3, max_length=300)


@app.get("/")
def pagina():
    """Entrega la interfaz web (HTML/CSS/JS)."""
    return FileResponse(RUTA_WEB)


@app.get("/salud")
def salud():
    """Permite comprobar que el servidor, el modelo y el LLM están listos."""
    return {"estado": "ok", "modelo": RUTA_MODELO.name, "filtro": CONFIG["filtro"], "umbral": UMBRAL,
            "llm_cargado": llm_cargado(), "modelo_llm": recomendaciones_llm.MODELO_LLM,
            "modelo_llm_segundo_intento": recomendaciones_llm.MODELO_LIGERO,
            "gradcam": CABEZA_GRADCAM is not None,
            "filtro_clip": FILTRO_CLIP is not None, "umbral_clip": UMBRAL_CLIP,
            "tabla_q": AGENTE.almacen.descripcion,
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
        gris = bytes_a_gris(contenido, CONFIG)
        x = gris_a_tensor(gris, FILTRO)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 0) Filtro CLIP: si la imagen no parece una radiografía de tórax, no se analiza (422).
    revision_clip = None
    if FILTRO_CLIP is not None:
        try:
            revision_clip = FILTRO_CLIP.evaluar(contenido)
        except Exception as err:   # si CLIP falla, la imagen pasa
            print(f"Filtro CLIP falló, se deja pasar la imagen: {type(err).__name__}: {err}")
        if revision_clip and not revision_clip["es_radiografia"]:
            print(f"Imagen rechazada por CLIP: P(radiografía)={revision_clip['prob_radiografia']}")
            raise HTTPException(status_code=422, detail=filtro_clip.MENSAJE_RECHAZO)

    prob, activaciones = MODELO.predecir(x)          # [P(NORMAL), P(PNEUMONIA)], ya es softmax
    prob_normal, prob_neumonia = float(prob[0]), float(prob[1])
    diagnostico = CLASES[1] if prob_neumonia >= UMBRAL else CLASES[0]

    # 1b) Grad-CAM de la clase predicha. Si falla, se responde igual, sin mapa de calor.
    mapa = None
    if CABEZA_GRADCAM is not None:
        try:
            cam, _ = CABEZA_GRADCAM.mapa(activaciones, CLASES.index(diagnostico))
            mapa = gradcam.imagenes(cam, gris)
        except Exception as err:
            print(f"Grad-CAM falló: {type(err).__name__}: {err}")

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
        "gradcam": mapa,   # PNG en base64 (gradcam_png = mezcla con 40 % de mapa) o None
        "filtro_clip": revision_clip,   # probabilidades de CLIP, o None si el filtro no está activo
    }


@app.post("/rl/feedback")
def feedback(f: Feedback):
    """El usuario califica la recomendación y el agente actualiza su tabla Q."""
    try:
        datos = AGENTE.registrar_feedback(f.estado, f.accion, recompensa=f.recompensa, estrellas=f.estrellas)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as err:   # la base de datos no respondió: no se pierde nada, el usuario puede reintentar
        print(f"Error guardando la tabla Q: {err}")
        raise HTTPException(status_code=503, detail="No se pudo guardar la calificación. Inténtalo de nuevo.")
    return {"estado": "ok", "mensaje": "Gracias, su calificación actualizó al agente.", "datos": datos}


@app.get("/asistente/preguntas")
def preguntas_sugeridas():
    """Preguntas sugeridas que muestra la web (también se puede escribir cualquier otra)."""
    return {"preguntas": list(asistente.PREGUNTAS_SUGERIDAS.values())}


@app.post("/asistente")
def preguntar(p: Pregunta):
    """Responde una pregunta del usuario con el LLM, usando el diagnóstico y la probabilidad como contexto."""
    return asistente.responder(p.diagnostico, p.prob_neumonia, p.pregunta, METRICAS)


@app.get("/rl/estado")
def estado_rl():
    """Muestra lo que ha aprendido el agente (tabla Q, epsilon y feedback recibido)."""
    return AGENTE.resumen()


@app.get("/rl/historial")
def historial_rl():
    """Historial de calificaciones con el promedio móvil de las últimas 10 recompensas (para la gráfica)."""
    return AGENTE.historial_con_promedio()
