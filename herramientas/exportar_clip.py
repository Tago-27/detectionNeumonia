"""
Prepara el filtro "¿es una radiografía de tórax?" con CLIP (openai/clip-vit-base-patch32), zero-shot.

Por qué así: CLIP con transformers + torch necesita más de 1 GB de RAM y Render gratis solo da 512 MB. Se exporta
solo el codificador de IMAGEN a ONNX y se cuantiza a int8 (~90 MB); los vectores de los 6 textos se calculan aquí
una sola vez y se guardan, así la API solo necesita onnxruntime.

Genera:
  modelo/clip_vision_int8.onnx   codificador de imagen + proyección (vector de 512, normalizado)
  modelo/clip_textos.npz         textos, sus vectores normalizados, escala de los logits y normalización de la imagen

Uso (desde la raíz del proyecto; torch y transformers solo se necesitan en local):
    pip install torch transformers onnx onnxruntime
    python herramientas/exportar_clip.py imagen1.jpg imagen2.png ...

Con las imágenes indicadas compara el resultado de la API (preprocesamiento con OpenCV + ONNX int8) con el de CLIP
original en PyTorch.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

BASE = Path(__file__).resolve().parent.parent
CARPETA_MODELO = BASE / "modelo"
sys.path.insert(0, str(BASE / "api"))
from filtro_clip import TEXTOS, FiltroCLIP  # noqa: E402

NOMBRE = "openai/clip-vit-base-patch32"
RUTA_FP32 = CARPETA_MODELO / "clip_vision_fp32.onnx"   # intermedio, se borra al final
RUTA_ONNX = CARPETA_MODELO / "clip_vision_int8.onnx"
RUTA_TEXTOS = CARPETA_MODELO / "clip_textos.npz"


class CodificadorImagen(torch.nn.Module):
    def __init__(self, modelo: CLIPModel):
        super().__init__()
        self.vision, self.proyeccion = modelo.vision_model, modelo.visual_projection

    def forward(self, pixel_values):
        vector = self.proyeccion(self.vision(pixel_values=pixel_values).pooler_output)
        return vector / vector.norm(dim=-1, keepdim=True)


def main(imagenes: list) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"Cargando {NOMBRE} ...")
    modelo = CLIPModel.from_pretrained(NOMBRE).eval()
    procesador = CLIPProcessor.from_pretrained(NOMBRE)

    # 1) Vectores de los textos (se calculan una sola vez).
    with torch.no_grad():
        tokens = procesador.tokenizer(TEXTOS, padding=True, return_tensors="pt")
        texto = modelo.text_projection(modelo.text_model(**tokens).pooler_output)
        texto = (texto / texto.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
        escala = float(modelo.logit_scale.exp())
    ip = procesador.image_processor
    np.savez(RUTA_TEXTOS, textos=np.array(TEXTOS), vectores=texto, escala=escala,
             media=np.array(ip.image_mean, np.float32), desviacion=np.array(ip.image_std, np.float32),
             tamano=int(ip.crop_size["height"]))
    print(f"Guardado {RUTA_TEXTOS.name}: {len(TEXTOS)} textos, escala {escala:.2f}")

    # 2) Codificador de imagen -> ONNX -> int8.
    ejemplo = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(CodificadorImagen(modelo), (ejemplo,), str(RUTA_FP32), input_names=["pixel_values"],
                      output_names=["vector"], dynamic_axes={"pixel_values": {0: "lote"}, "vector": {0: "lote"}},
                      opset_version=17, dynamo=False)
    quantize_dynamic(str(RUTA_FP32), str(RUTA_ONNX), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"])
    RUTA_FP32.unlink()
    for extra in CARPETA_MODELO.glob("clip_vision_fp32.onnx*"):
        extra.unlink()
    print(f"Guardado {RUTA_ONNX.name} ({RUTA_ONNX.stat().st_size / 2**20:.1f} MB)")

    if not imagenes:
        print("AVISO: no se pasaron imágenes, así que no se comparó con CLIP original.")
        return

    # 3) Verificación: lo que hará la API frente a CLIP original.
    from PIL import Image
    filtro = FiltroCLIP(RUTA_ONNX, RUTA_TEXTOS, umbral=0.5)
    diferencia = 0.0
    for ruta in imagenes:
        with torch.no_grad():
            entrada = procesador(text=TEXTOS, images=Image.open(ruta).convert("RGB"), return_tensors="pt", padding=True)
            p_torch = modelo(**entrada).logits_per_image.softmax(dim=-1)[0].numpy()
        p_api = filtro.evaluar(Path(ruta).read_bytes())["probabilidades"]
        p_api = np.array([p_api[t] for t in TEXTOS])
        diferencia = max(diferencia, float(np.abs(p_torch - p_api).max()))
        print(f"  {Path(ruta).name}: P(radiografía) CLIP original={p_torch[0]:.3f}  API (ONNX int8)={p_api[0]:.3f}  "
              f"texto más probable: '{TEXTOS[int(p_api.argmax())]}'")
    print(f"Diferencia máxima de probabilidades: {diferencia:.3f}")
    if diferencia > 0.1:
        sys.exit("ERROR: el filtro de la API se aleja demasiado de CLIP original.")
    print("OK: el filtro de la API reproduce a CLIP original.")


if __name__ == "__main__":
    main(sys.argv[1:])
