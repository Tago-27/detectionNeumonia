"""
Recomendaciones generadas con un LLM a partir de la predicción del modelo de neumonía.

Pipeline:  predicción (diagnóstico + probabilidad) → prompt → LLM (Gemini) → filtro → recomendación
- La predicción entra como INPUT del prompt.
- El LLM es Google Gemini, llamado por su API (plan gratuito): un LLM local no cabe en los 512 MB de Render gratis.
- El estilo de la recomendación lo elige el agente de aprendizaje por refuerzo.
- Si el LLM falla o su texto no pasa el filtro, se usa un texto de respaldo (también de más de 200 caracteres).

Variables de entorno:
  GEMINI_API_KEY    clave gratuita de https://aistudio.google.com/apikey (sin ella se usan los textos de respaldo)
  MODELO_LLM        modelo de Gemini (opcional). Si no se indica o no está disponible para la clave, al arrancar
                    se elige solo el Gemini Flash más reciente de la lista de modelos de la cuenta.
  USAR_LLM=0        desactiva el LLM (pruebas sin internet)
"""
import os
import re
from typing import Dict, List, Optional

MIN_CARACTERES = 200
AVISO = "Resultado experimental, no es un diagnóstico médico. Ante cualquier duda, consulte a un profesional de la salud."

CONSEJOS = {
    "PNEUMONIA": [
        "Consultar a un médico lo antes posible para confirmar el resultado con un examen clínico.",
        "Acudir a urgencias si hay dificultad para respirar, respiración muy rápida, labios o uñas azulados, "
        "dolor en el pecho, fiebre muy alta o que no baja, somnolencia o confusión.",
        "En niños, son señales de alarma el hundimiento de las costillas al respirar, el quejido y el rechazo a comer o beber.",
        "No automedicarse ni tomar antibióticos sin fórmula médica.",
        "Mantener reposo e hidratación, cubrirse al toser y lavarse las manos con frecuencia.",
    ],
    "NORMAL": [
        "El resultado no muestra signos compatibles con neumonía, pero no descarta otras enfermedades.",
        "Consultar a un médico si hay tos persistente, fiebre, dolor en el pecho o dificultad para respirar.",
        "Mantener al día las vacunas recomendadas por el médico, como influenza y neumococo.",
        "Lavarse las manos con frecuencia y evitar el humo de tabaco.",
        "Una alimentación balanceada, buen descanso y actividad física ayudan a mantener las defensas.",
    ],
}

# Estilos de recomendación: son las ACCIONES que elige el agente de refuerzo.
ESTILOS = {
    "breve_directa": "Escribe una recomendación directa y práctica, en un solo párrafo de 3 o 4 oraciones.",
    "detallada": "Escribe una recomendación completa y ordenada, explicando el porqué de cada sugerencia, en 5 o 6 oraciones.",
    "alarma_prevencion": "Escribe una recomendación que destaque primero las señales de alarma o las medidas de prevención, en 4 o 5 oraciones.",
    "empatica": "Escribe la recomendación en un tono cálido y comprensivo, dirigiéndote al paciente, en 4 o 5 oraciones.",
}

_PROHIBIDAS = re.compile(r"\b(mg|ml|dosis|miligramos|amoxicilina|azitromicina|ibuprofeno|acetaminof[eé]n|paracetamol|"
                         r"garantiz\w*|curar\w*|seguro que)\b", re.IGNORECASE)
_SALUDO = re.compile(r"^\W*(hola|buen[oa]s)\b|estoy aqu[ií] para", re.IGNORECASE)
MODELO_LLM: Optional[str] = os.getenv("MODELO_LLM") or None   # se completa en cargar_llm()
# Variantes que no sirven para generar texto normal (o son experimentales) y no se eligen automáticamente.
_VARIANTES_EXCLUIDAS = ("lite", "image", "tts", "audio", "live", "embedding", "thinking", "exp", "preview", "native")
INSTRUCCIONES_SISTEMA = ("Redactas recomendaciones de salud en español para pacientes. Escribes solo el texto de la "
                         "recomendación, en prosa, sin saludos, sin preguntas, sin listas, sin títulos y sin formato "
                         "markdown. No inventas información: solo reformulas las ideas que te dan. "
                         "No mencionas medicamentos ni dosis.")
_cliente = None
_motivo_sin_llm: Optional[str] = None   # por qué no hay LLM (se muestra en la web junto al texto de respaldo)


def _version(nombre: str) -> float:
    encontrada = re.search(r"gemini-(\d+(?:\.\d+)?)", nombre)
    return float(encontrada.group(1)) if encontrada else 0.0


def elegir_modelo(disponibles: List[str], preferido: Optional[str]) -> Optional[str]:
    """Usa el modelo pedido si existe; si no, el Gemini Flash estable más reciente de la lista."""
    nombres = [n.removeprefix("models/") for n in disponibles]
    if preferido and preferido.removeprefix("models/") in nombres:
        return preferido.removeprefix("models/")
    flash = [n for n in nombres if n.startswith("gemini-") and "flash" in n]
    estables = [n for n in flash if not any(v in n for v in _VARIANTES_EXCLUIDAS)]
    candidatos = estables or flash
    # Mayor versión primero; a igual versión, el nombre más corto (gemini-2.5-flash antes que gemini-2.5-flash-001).
    return max(candidatos, key=lambda n: (_version(n), -len(n))) if candidatos else None


def cargar_llm() -> None:
    """Prepara el cliente de Gemini y elige el modelo una sola vez, al arrancar la API."""
    global _cliente, MODELO_LLM, _motivo_sin_llm
    if os.getenv("USAR_LLM", "1") != "1":
        _motivo_sin_llm = "LLM desactivado (USAR_LLM=0)"
    elif not os.getenv("GEMINI_API_KEY"):
        _motivo_sin_llm = "LLM no configurado (falta GEMINI_API_KEY)"
    if _motivo_sin_llm:
        print(f"Advertencia: {_motivo_sin_llm}. Se usarán textos de respaldo.")
        return
    from google import genai
    from google.genai import types
    # timeout en milisegundos: si Gemini tarda demasiado se responde con el texto de respaldo.
    cliente = genai.Client(http_options=types.HttpOptions(timeout=25_000))
    # Google renombra sus modelos a menudo: se consulta la lista real de la cuenta en vez de fijar un nombre.
    try:
        disponibles = [m.name for m in cliente.models.list()
                       if not m.supported_actions or "generateContent" in m.supported_actions]
    except Exception as exc:
        _motivo_sin_llm = f"no se pudo conectar con Gemini al arrancar: {str(exc)[:200]}"
        print(f"Advertencia: {_motivo_sin_llm}. Se usarán textos de respaldo.")
        return
    elegido = elegir_modelo(disponibles, MODELO_LLM)
    if not elegido:
        _motivo_sin_llm = "la clave de Gemini no tiene modelos Flash disponibles"
        print(f"Advertencia: {_motivo_sin_llm} ({disponibles[:10]}).")
        return
    if MODELO_LLM and elegido != MODELO_LLM.removeprefix("models/"):
        print(f"Advertencia: MODELO_LLM='{MODELO_LLM}' no está disponible; se usa '{elegido}'.")
    MODELO_LLM, _cliente = elegido, cliente
    print(f"LLM listo: {MODELO_LLM} (API de Gemini)")


def llm_cargado() -> bool:
    return _cliente is not None


def construir_prompt(diagnostico: str, prob_neumonia: float, estilo: str) -> str:
    """La predicción de la CNN (diagnóstico y probabilidad) es el INPUT del LLM."""
    resultado = "signos compatibles con neumonía" if diagnostico == "PNEUMONIA" else "sin signos compatibles con neumonía"
    pct = round(prob_neumonia * 100, 1)
    ideas = "\n".join(f"- {c}" for c in CONSEJOS[diagnostico])
    return (f"Resultado de una radiografía de tórax analizada por IA: {resultado} "
            f"(probabilidad de neumonía: {pct} %).\n\nIdeas que debes usar:\n{ideas}\n\n"
            f"{ESTILOS[estilo]} Empieza mencionando el resultado y termina recordando que debe confirmarlo un médico.")


def _llamar_llm(prompt: str) -> str:
    from google.genai import types
    respuesta = _cliente.models.generate_content(
        model=MODELO_LLM,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=INSTRUCCIONES_SISTEMA, temperature=0.7,
                                           max_output_tokens=2048),
    )
    return respuesta.text or ""


def _limpiar(texto: str) -> str:
    """Quita títulos/markdown y, si el LLM se cortó por max_new_tokens, la última frase a medias."""
    texto = re.sub(r"^\s*\**\s*recomendaci[oó]n\s*:?\s*\**\s*:?\s*", "", texto.strip(), flags=re.IGNORECASE)
    texto = texto.replace("**", "").strip()   # sin formato markdown: la web lo muestra como texto plano
    texto = re.sub(r"^¡?\s*(hola|buen[oa]s( d[ií]as| tardes)?)\s*[!,.]*\s*", "", texto, flags=re.IGNORECASE)
    texto = texto[:1].upper() + texto[1:]
    if texto.endswith((".", "!")):
        return texto
    fin = texto.rfind(".")
    return texto[:fin + 1] if fin != -1 else texto


def _motivo_rechazo(texto: str) -> Optional[str]:
    """Filtro de seguridad y calidad del texto generado. Devuelve el motivo si se rechaza, o None si es válido."""
    if len(texto) < MIN_CARACTERES:
        return f"texto corto ({len(texto)} caracteres)"
    prohibida = _PROHIBIDAS.search(texto)
    if prohibida:
        return f"palabra no permitida: '{prohibida.group(0)}'"
    if _SALUDO.search(texto):
        return "respuesta conversacional en vez de recomendación"
    return None


def texto_respaldo(diagnostico: str, prob_neumonia: float) -> str:
    pct = round(prob_neumonia * 100, 1)
    inicio = (f"El modelo encontró signos compatibles con neumonía, con una probabilidad de {pct} %. "
              if diagnostico == "PNEUMONIA" else
              f"El modelo no encontró signos compatibles con neumonía (probabilidad de neumonía: {pct} %). ")
    return inicio + " ".join(CONSEJOS[diagnostico][:3]) + " Este resultado debe confirmarlo un médico."


def generar_recomendacion(diagnostico: str, prob_neumonia: float, estilo: str) -> Dict[str, object]:
    texto: Optional[str] = None
    motivo: Optional[str] = None   # por qué se usó el texto de respaldo (queda en los logs y en el JSON)
    if _cliente is None:
        motivo = _motivo_sin_llm or "LLM no disponible"
    else:
        try:
            texto = _limpiar(_llamar_llm(construir_prompt(diagnostico, prob_neumonia, estilo)))
            motivo = "respuesta vacía del LLM" if not texto else _motivo_rechazo(texto)
        except Exception as exc:   # sin red, clave inválida, cuota gratuita agotada (429), timeout...
            motivo = f"error del LLM: {type(exc).__name__}: {str(exc)[:200]}"
        if motivo:
            print(f"Texto de respaldo -> {motivo}")
            texto = None
    generado = texto is not None
    return {
        "recomendacion": texto if generado else texto_respaldo(diagnostico, prob_neumonia),
        "generado_por_llm": generado,
        "motivo_respaldo": motivo,
        "modelo_llm": MODELO_LLM if generado else None,
        "estilo": estilo,
        "consejos": CONSEJOS[diagnostico],
        "aviso": AVISO,
    }
