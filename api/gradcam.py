"""
Grad-CAM: mapa de calor de las zonas de la radiografía en las que se fija la CNN.

  1. Activaciones A de la última capa convolucional (relu6b, 7x7x512): las da el propio .tflite como segunda salida.
  2. Gradiente de la probabilidad de la clase predicha respecto a A.
  3. Peso de cada canal = promedio de su gradiente; mapa = ReLU(suma ponderada de los canales), normalizado de 0 a 1.
  4. Se redimensiona a 224x224, se colorea con COLORMAP_JET y se mezcla con la radiografía (40 % mapa).

Por qué el gradiente se calcula con numpy: TensorFlow completo no cabe en Render gratis (512 MB) y el intérprete
TFLite no calcula gradientes. Detrás de relu6b solo hay MaxPooling, Flatten y tres capas densas, así que la
retropropagación se puede escribir a mano con sus pesos (modelo/*.npz, exportados por herramientas/convertir_a_tflite.py).
Ese script comprueba que el resultado coincide con el Grad-CAM calculado con tf.GradientTape.
"""
import base64
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

OPACIDAD_MAPA = 0.4


class CabezaGradCAM:
    """Las capas que van después de relu6b: MaxPooling2D -> Flatten -> Dense relu -> Dense relu -> Dense softmax."""

    def __init__(self, ruta: Path):
        datos = np.load(ruta)
        self.pool, self.paso = int(datos["pool"]), int(datos["paso"])
        self.capas = [(datos[f"W{i}"].astype(np.float32), datos[f"b{i}"].astype(np.float32)) for i in (1, 2, 3)]

    def mapa(self, activaciones: np.ndarray, clase: int) -> Tuple[np.ndarray, np.ndarray]:
        """activaciones: (alto, ancho, canales) de relu6b. Devuelve (mapa 0..1 del tamaño de A, probabilidades)."""
        A = activaciones.astype(np.float32)
        alto, ancho, canales = A.shape
        p, s = self.pool, self.paso
        alto_p, ancho_p = (alto - p) // s + 1, (ancho - p) // s + 1   # MaxPooling con padding "valid"
        canal = np.arange(canales)

        # --- Hacia adelante (guardando qué posición ganó cada ventana del MaxPooling)
        pooled = np.empty((alto_p, ancho_p, canales), np.float32)
        ganador = np.empty((alto_p, ancho_p, canales), np.int64)
        for i in range(alto_p):
            for j in range(ancho_p):
                ventana = A[i * s:i * s + p, j * s:j * s + p].reshape(p * p, canales)
                ganador[i, j] = ventana.argmax(axis=0)
                pooled[i, j] = ventana[ganador[i, j], canal]
        (W1, b1), (W2, b2), (W3, b3) = self.capas
        z1 = pooled.reshape(-1) @ W1 + b1          # Flatten (orden alto, ancho, canal, como Keras) + densa1
        h1 = np.maximum(z1, 0)
        z2 = h1 @ W2 + b2                          # densa2
        h2 = np.maximum(z2, 0)
        z3 = h2 @ W3 + b3                          # salida (softmax)
        e = np.exp(z3 - z3.max())
        prob = e / e.sum()

        # --- Hacia atrás: d prob[clase] / d A
        uno = np.zeros_like(prob)
        uno[clase] = 1.0
        g = prob[clase] * (uno - prob)             # derivada del softmax
        g = (W3 @ g) * (z2 > 0)
        g = (W2 @ g) * (z1 > 0)
        g_pooled = (W1 @ g).reshape(alto_p, ancho_p, canales)
        g_A = np.zeros_like(A)                     # el gradiente solo pasa por la posición ganadora de cada ventana
        for i in range(alto_p):
            for j in range(ancho_p):
                fila, col = ganador[i, j] // p, ganador[i, j] % p
                g_A[i * s + fila, j * s + col, canal] += g_pooled[i, j]

        # --- Grad-CAM
        pesos = g_A.mean(axis=(0, 1))              # promedio del gradiente por canal
        cam = np.maximum(A @ pesos, 0)             # suma ponderada + ReLU
        maximo = cam.max()
        return (cam / maximo if maximo > 0 else cam), prob


def _png_base64(imagen: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", imagen)
    if not ok:
        raise ValueError("No se pudo codificar el PNG.")
    return base64.b64encode(png.tobytes()).decode("ascii")


def imagenes(cam: np.ndarray, gris: np.ndarray) -> dict:
    """PNG en base64: radiografía (como la ve el modelo), mapa de calor solo y mezcla con 40 % de mapa."""
    alto, ancho = gris.shape
    mapa = cv2.resize(cam.astype(np.float32), (ancho, alto), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap(np.uint8(np.clip(mapa, 0, 1) * 255), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    mezcla = cv2.addWeighted(color, OPACIDAD_MAPA, base, 1 - OPACIDAD_MAPA, 0)
    return {"gradcam_png": _png_base64(mezcla), "mapa_calor_png": _png_base64(color),
            "radiografia_png": _png_base64(gris), "opacidad_mapa": OPACIDAD_MAPA}
