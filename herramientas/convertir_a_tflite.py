"""
Convierte el modelo .keras aprobado a .tflite para el despliegue.

Por qué: TensorFlow completo ocupa ~565 MB de RAM y Render gratis solo da 512 MB.
El intérprete LiteRT (ai-edge-litert) ejecuta el .tflite con una fracción de esa memoria
y da las mismas probabilidades (diferencia medida: ~1e-7).

Uso (desde la raíz del proyecto; TensorFlow solo se necesita en local, no en Render):
    pip install tensorflow==2.20.0 keras==3.15.1
    python herramientas/convertir_a_tflite.py radiografia1.jpg radiografia2.png ...

Genera modelo/<mismo nombre>.tflite y comprueba con las radiografías indicadas (con el mismo
preprocesamiento de la API) que da las mismas probabilidades que el .keras.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE = Path(__file__).resolve().parent.parent
CARPETA_MODELO = BASE / "modelo"
sys.path.insert(0, str(BASE / "api"))
from preprocesamiento import bytes_a_tensor, construir_filtro  # noqa: E402


# Métrica personalizada del entrenamiento: necesaria para poder cargar el .keras.
@keras.utils.register_keras_serializable(package="neumonia")
class AUCPneumonia(keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred[:, 1], sample_weight)


def unico(patron: str) -> Path:
    encontrados = sorted(CARPETA_MODELO.glob(patron))
    if len(encontrados) != 1:
        sys.exit(f"Debe haber exactamente un '{patron}' en {CARPETA_MODELO} (hay {len(encontrados)}).")
    return encontrados[0]


def main(radiografias: list) -> None:
    origen = unico("*.keras")
    destino = origen.with_suffix(".tflite")
    config = json.loads(unico("config*.json").read_text(encoding="utf-8"))

    print(f"Cargando {origen.name} ...")
    modelo = keras.models.load_model(origen, compile=False)

    # Función de inferencia con forma fija (1, 224, 224, 1) y training=False: las capas de augmentation no actúan.
    firma = tf.TensorSpec([1, config["img_h"], config["img_w"], 1], tf.float32, name="imagen")
    funcion = tf.function(lambda x: modelo(x, training=False), input_signature=[firma])
    convertidor = tf.lite.TFLiteConverter.from_concrete_functions([funcion.get_concrete_function()], modelo)
    destino.write_bytes(convertidor.convert())
    print(f"Guardado {destino.name} ({destino.stat().st_size / 2**20:.1f} MB)")

    if not radiografias:
        print("AVISO: no se pasaron radiografías, así que no se verificó el .tflite. "
              "Repite indicando algunas imágenes (con ruido sintético el modelo siempre satura en 1,0).")
        return

    from ai_edge_litert.interpreter import Interpreter
    interprete = Interpreter(model_path=str(destino))
    interprete.allocate_tensors()
    ent, sal = interprete.get_input_details()[0], interprete.get_output_details()[0]
    filtro = construir_filtro(config)
    diferencia = 0.0
    for ruta in radiografias:
        x = bytes_a_tensor(Path(ruta).read_bytes(), config, filtro).astype(np.float32)
        p_keras = modelo.predict(x, verbose=0)[0]
        interprete.set_tensor(ent["index"], x)
        interprete.invoke()
        p_lite = interprete.get_tensor(sal["index"])[0]
        diferencia = max(diferencia, float(np.abs(p_keras - p_lite).max()))
        print(f"  {Path(ruta).name}: keras P(neumonía)={p_keras[1]:.6f}   tflite={p_lite[1]:.6f}")
    print(f"Diferencia máxima: {diferencia:.2e}")
    if diferencia > 1e-3:
        sys.exit("ERROR: el .tflite no reproduce al .keras. No lo despliegues.")
    print("OK: el .tflite reproduce las probabilidades del .keras.")


if __name__ == "__main__":
    main(sys.argv[1:])
