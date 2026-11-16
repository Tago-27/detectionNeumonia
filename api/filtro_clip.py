"""
Filtro "¿es una radiografía de tórax?" con CLIP (openai/clip-vit-base-patch32), zero-shot.

La imagen se compara con 6 descripciones; si la probabilidad de "a chest x-ray radiograph" es menor que el umbral,
la API responde 422. Solo se usa el codificador de imagen de CLIP, en ONNX cuantizado a int8, con onnxruntime;
los vectores de los textos ya están calculados en modelo/clip_textos.npz (herramientas/exportar_clip.py).
Si CLIP no se puede cargar, la API deja pasar todas las imágenes.
"""
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

TEXTOS = [
    "a chest x-ray radiograph",
    "a photo of an animal",
    "a photo of a person",
    "a photo of an object",
    "a photo of a landscape",
    "a document with text",
]
MENSAJE_RECHAZO = "La imagen no parece una radiografía de tórax"


class FiltroCLIP:
    def __init__(self, ruta_onnx: Path, ruta_textos: Path, umbral: float = 0.5):
        import onnxruntime as ort
        opciones = ort.SessionOptions()
        opciones.intra_op_num_threads = 1
        opciones.inter_op_num_threads = 1
        opciones.enable_cpu_mem_arena = False   # menos RAM reservada (Render gratis: 512 MB)
        self._sesion = ort.InferenceSession(str(ruta_onnx), opciones, providers=["CPUExecutionProvider"])
        datos = np.load(ruta_textos)
        if list(datos["textos"]) != TEXTOS:
            raise ValueError("Los textos de clip_textos.npz no coinciden con TEXTOS: vuelve a ejecutar exportar_clip.py")
        self._vectores = datos["vectores"].astype(np.float32)
        self._escala = float(datos["escala"])
        self._media = datos["media"].astype(np.float32)
        self._desviacion = datos["desviacion"].astype(np.float32)
        self._tamano = int(datos["tamano"])
        self.umbral = umbral

    def _preprocesar(self, contenido: bytes) -> np.ndarray:
        """Igual que CLIPImageProcessor: lado corto a 224 (bicúbica), recorte central 224x224, normalización."""
        img = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("imagen no válida")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alto, ancho = img.shape[:2]
        t = self._tamano
        escala = t / min(alto, ancho)
        nuevo = (max(t, round(ancho * escala)), max(t, round(alto * escala)))
        # Al reducir se usa INTER_AREA (equivale al antialias de PIL); al ampliar, bicúbica.
        img = cv2.resize(img, nuevo, interpolation=cv2.INTER_AREA if escala < 1 else cv2.INTER_CUBIC)
        y, x = (img.shape[0] - t) // 2, (img.shape[1] - t) // 2
        img = img[y:y + t, x:x + t].astype(np.float32) / 255.0
        img = (img - self._media) / self._desviacion
        return img.transpose(2, 0, 1)[np.newaxis]   # (1, 3, 224, 224)

    def evaluar(self, contenido: bytes) -> Dict[str, object]:
        vector = self._sesion.run(None, {"pixel_values": self._preprocesar(contenido)})[0][0]
        logits = self._escala * (self._vectores @ vector)
        e = np.exp(logits - logits.max())
        prob = e / e.sum()
        p_rx = float(prob[0])
        return {"prob_radiografia": round(p_rx, 4), "umbral": self.umbral, "es_radiografia": p_rx >= self.umbral,
                "probabilidades": {t: round(float(p), 4) for t, p in zip(TEXTOS, prob)}}
