"""
Preprocesamiento de la API.

Debe ser IDÉNTICO al usado en el entrenamiento:
  1. leer en escala de grises
  2. redimensionar a 224x224 con cv2.INTER_AREA
  3. aplicar el filtro elegido en validación (guardado en el config*.json del modelo)
  4. dar forma (1, 224, 224, 1)

La normalización 1/255 NO se hace aquí: ya está dentro del modelo (capa Rescaling).
"""
import cv2
import numpy as np


def _conv(img, kernel):
    r = cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(r, 0, 255).astype(np.uint8)


def construir_filtro(config):
    """Devuelve una función img2d -> img2d según el filtro guardado en la configuración."""
    k = {nombre: np.array(valores, dtype=np.float32) for nombre, valores in config["kernels"].items()}
    clip = config["clahe"]["clip_limit"]
    tile = tuple(config["clahe"]["tile_grid"])

    def clahe(img):
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(img)

    filtros = {
        "original": lambda im: im,
        "sharpen": lambda im: _conv(im, k["sharpen"]),
        "edge_detect": lambda im: _conv(im, k["edge"]),
        "strong_edge_detect": lambda im: _conv(im, k["strong_edge"]),
        "clahe": clahe,
        "clahe_sharpen": lambda im: _conv(clahe(im), k["sharpen"]),
    }
    nombre = config["filtro"]
    if nombre not in filtros:
        raise ValueError(f"Filtro desconocido en la configuración: {nombre}")
    return filtros[nombre]


def bytes_a_tensor(contenido: bytes, config, filtro):
    """Convierte los bytes de la imagen subida en el arreglo que espera el modelo."""
    arreglo = np.frombuffer(contenido, dtype=np.uint8)
    img = cv2.imdecode(arreglo, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("El archivo no es una imagen válida (usa JPG o PNG).")
    img = cv2.resize(img, (config["img_w"], config["img_h"]), interpolation=cv2.INTER_AREA)
    img = filtro(img)
    return img[np.newaxis, :, :, np.newaxis]   # (1, alto, ancho, 1)
