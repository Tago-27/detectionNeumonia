"""
Convierte el modelo .keras aprobado a .tflite para el despliegue, preparado también para Grad-CAM.

Por qué: TensorFlow completo ocupa ~565 MB de RAM y Render gratis solo da 512 MB.
El intérprete LiteRT (ai-edge-litert) ejecuta el .tflite con una fracción de esa memoria
y da las mismas probabilidades (diferencia medida: ~1e-7).

Para Grad-CAM se genera:
  - modelo/<nombre>.tflite con DOS salidas: activaciones de la última capa convolucional (relu6b) y probabilidades;
  - modelo/<nombre>_gradcam.npz con los pesos de las capas que van después (MaxPooling + 3 densas), para que la API
    calcule el gradiente con numpy (api/gradcam.py) sin necesitar TensorFlow.

Uso (desde la raíz del proyecto; TensorFlow solo se necesita en local, no en Render):
    pip install tensorflow==2.20.0 keras==3.15.1
    python herramientas/convertir_a_tflite.py radiografia1.jpg radiografia2.png ...

Con las radiografías indicadas (mismo preprocesamiento de la API) comprueba que el .tflite da las mismas
probabilidades que el .keras y que el Grad-CAM de numpy coincide con el calculado con tf.GradientTape.
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
from gradcam import CabezaGradCAM  # noqa: E402
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


def exportar_cabeza(modelo, capa_conv: str, destino: Path) -> None:
    """Guarda los pesos de las capas que siguen a la última convolucional. Comprueba que la arquitectura es la que
    api/gradcam.py sabe derivar a mano: MaxPooling2D -> Dropout -> Flatten -> (Dense relu, Dropout) x2 -> Dense softmax."""
    capas = modelo.layers[[c.name for c in modelo.layers].index(capa_conv) + 1:]
    utiles = [c for c in capas if not isinstance(c, keras.layers.Dropout)]   # Dropout no actúa al predecir
    tipos = [type(c).__name__ for c in utiles]
    if tipos != ["MaxPooling2D", "Flatten", "Dense", "Dense", "Dense"]:
        sys.exit(f"ERROR: después de {capa_conv} hay {tipos}; api/gradcam.py no sabe derivar esa cabeza.")
    pool, _, d1, d2, d3 = utiles
    activaciones = [d.activation.__name__ for d in (d1, d2, d3)]
    if activaciones != ["relu", "relu", "softmax"] or pool.padding != "valid":
        sys.exit(f"ERROR: activaciones {activaciones} / padding {pool.padding} no soportados por api/gradcam.py.")
    if pool.pool_size[0] != pool.pool_size[1] or pool.strides[0] != pool.strides[1]:
        sys.exit("ERROR: el MaxPooling debe ser cuadrado.")
    pesos = {}
    for i, d in enumerate((d1, d2, d3), start=1):
        pesos[f"W{i}"], pesos[f"b{i}"] = [w.astype(np.float32) for w in d.get_weights()]
    np.savez_compressed(destino, pool=pool.pool_size[0], paso=pool.strides[0], **pesos)
    print(f"Guardado {destino.name} ({destino.stat().st_size / 2**20:.1f} MB)")


def gradcam_tf(auxiliar, x: np.ndarray, clase: int) -> np.ndarray:
    """Grad-CAM de referencia con tf.GradientTape (lo que la API reproduce con numpy)."""
    with tf.GradientTape() as cinta:
        A, prob = auxiliar(x, training=False)
        puntaje = prob[:, clase]
    gradientes = cinta.gradient(puntaje, A)
    pesos = tf.reduce_mean(gradientes, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(A[0] * pesos, axis=-1)).numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def main(radiografias: list) -> None:
    origen = unico("*.keras")
    destino = origen.with_suffix(".tflite")
    destino_npz = origen.with_name(origen.stem + "_gradcam.npz")
    config = json.loads(unico("config*.json").read_text(encoding="utf-8"))
    capa_conv = config.get("arquitectura", {}).get("ultima_capa_conv", "relu6b")

    print(f"Cargando {origen.name} ...")
    modelo = keras.models.load_model(origen, compile=False)
    auxiliar = keras.Model(modelo.inputs, [modelo.get_layer(capa_conv).output, modelo.output])

    # Función de inferencia con forma fija (1, 224, 224, 1) y training=False: las capas de augmentation no actúan.
    firma = tf.TensorSpec([1, config["img_h"], config["img_w"], 1], tf.float32, name="imagen")
    funcion = tf.function(lambda x: auxiliar(x, training=False), input_signature=[firma])
    convertidor = tf.lite.TFLiteConverter.from_concrete_functions([funcion.get_concrete_function()], auxiliar)
    destino.write_bytes(convertidor.convert())
    print(f"Guardado {destino.name} ({destino.stat().st_size / 2**20:.1f} MB), salidas: {capa_conv} + probabilidades")
    exportar_cabeza(modelo, capa_conv, destino_npz)

    if not radiografias:
        print("AVISO: no se pasaron radiografías, así que no se verificó el .tflite. "
              "Repite indicando algunas imágenes (con ruido sintético el modelo siempre satura en 1,0).")
        return

    from ai_edge_litert.interpreter import Interpreter
    interprete = Interpreter(model_path=str(destino))
    interprete.allocate_tensors()
    ent = interprete.get_input_details()[0]
    salidas = interprete.get_output_details()
    sal_prob = next(s for s in salidas if len(s["shape"]) == 2)
    sal_act = next(s for s in salidas if len(s["shape"]) == 4)
    cabeza = CabezaGradCAM(destino_npz)
    filtro = construir_filtro(config)
    dif_prob = dif_cam = 0.0
    for ruta in radiografias:
        x = bytes_a_tensor(Path(ruta).read_bytes(), config, filtro).astype(np.float32)
        p_keras = modelo.predict(x, verbose=0)[0]
        interprete.set_tensor(ent["index"], x)
        interprete.invoke()
        p_lite = interprete.get_tensor(sal_prob["index"])[0]
        A_lite = interprete.get_tensor(sal_act["index"])[0]
        clase = int(p_keras[1] >= config["umbral"])
        cam_numpy, p_numpy = cabeza.mapa(A_lite, clase)
        cam_ref = gradcam_tf(auxiliar, x, clase)
        dif_prob = max(dif_prob, float(np.abs(p_keras - p_lite).max()), float(np.abs(p_keras - p_numpy).max()))
        dif_cam = max(dif_cam, float(np.abs(cam_numpy - cam_ref).max()))
        print(f"  {Path(ruta).name}: keras P(neumonía)={p_keras[1]:.6f}   tflite={p_lite[1]:.6f}   "
              f"Grad-CAM numpy vs GradientTape={np.abs(cam_numpy - cam_ref).max():.2e}")
    print(f"Diferencia máxima: probabilidades {dif_prob:.2e}, Grad-CAM {dif_cam:.2e}")
    if dif_prob > 1e-3:
        sys.exit("ERROR: el .tflite no reproduce al .keras. No lo despliegues.")
    if dif_cam > 1e-2:
        sys.exit("ERROR: el Grad-CAM de numpy no coincide con el de TensorFlow. No lo despliegues.")
    print("OK: el .tflite reproduce las probabilidades del .keras y el Grad-CAM coincide con TensorFlow.")


if __name__ == "__main__":
    main(sys.argv[1:])
