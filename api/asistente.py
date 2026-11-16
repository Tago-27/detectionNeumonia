"""
Asistente de preguntas: el usuario pregunta sobre su resultado y el LLM responde con la predicción como contexto.

Pipeline:  diagnóstico + probabilidad + métricas del modelo + pregunta → prompt → LLM (Gemini) → filtro → respuesta
Reglas: español, máximo 4 oraciones, sin diagnosticar, sin medicamentos ni dosis y siempre sugerir consultar a un
médico. Si el LLM falla o la respuesta no pasa el filtro, se usa una respuesta de respaldo escrita a mano.
"""
import re
import unicodedata
from typing import Dict, Optional

import recomendaciones_llm as llm

MAX_ORACIONES = 4
MIN_CARACTERES = 40
AVISO = llm.AVISO

PREGUNTAS_SUGERIDAS = {
    "que_es": "¿Qué es la neumonía?",
    "urgencias": "¿Cuándo debo ir a urgencias?",
    "contagio": "¿Cómo se contagia y cómo evitarlo?",
    "que_hacer": "¿Qué debo hacer ahora con este resultado?",
    "confiable": "¿Qué tan confiable es este resultado?",
}

INSTRUCCIONES_ASISTENTE = (
    "Eres un asistente informativo de un proyecto académico que analiza radiografías de tórax con una red neuronal. "
    "Respondes en español, en prosa, en un máximo de 4 oraciones, sin saludos, sin listas y sin formato markdown. "
    "Nunca das un diagnóstico ni afirmas que la persona tiene o no tiene una enfermedad: el resultado del modelo es "
    "solo orientativo. Nunca mencionas medicamentos, antibióticos, nombres de fármacos ni dosis; si te preguntan por "
    "ellos, dices que eso solo lo puede indicar un médico. No prometes curas. Si la pregunta no trata sobre salud "
    "respiratoria, la neumonía o este resultado, respondes que solo puedes ayudar con esos temas. "
    "Terminas siempre sugiriendo consultar a un médico."
)

# Fármacos, dosis y formas farmacéuticas: si aparecen, la respuesta se descarta.
_FARMACOS = re.compile(
    r"\b(\d+([.,]\d+)?\s*(mg|ml|mcg|g|ui)\b|dosis|miligramos|mililitros|tabletas?|pastillas?|c[aá]psulas?|"
    r"jarabes?|inyecci\w*|antibi[oó]tic\w*|antiinflamatori\w*|antipir[eé]tic\w*|expectorant\w*|antitus\w*|"
    r"ibuprofeno|acetaminof[eé]n|paracetamol|aspirina|\w+cilina|\w+micina|\w+floxacino|\w+profeno)\b",
    re.IGNORECASE)
_PROMESAS = re.compile(r"\b(garantiz\w*|curar\w*|seguro que)\b", re.IGNORECASE)
# Afirmar que la persona tiene (o no) neumonía es diagnosticar; "si usted tiene fiebre..." sí está permitido.
_DIAGNOSTICA = re.compile(r"(?<!si )\b(usted|t[uú])\s+(no\s+)?(tiene|tienes|padece|padeces)\s+(una\s+)?neumon",
                          re.IGNORECASE)
_MEDICO = re.compile(r"m[eé]dic|profesional de la salud|urgencias", re.IGNORECASE)


def _normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", sin_tildes.lower()).strip()


_CLAVE_POR_PREGUNTA = {_normalizar(t): clave for clave, t in PREGUNTAS_SUGERIDAS.items()}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}".replace(".", ",") + " %"


def metricas_desde_config(config: dict) -> Dict[str, float]:
    """Datos de confiabilidad del modelo aprobado (vienen del config del entrenamiento)."""
    val, test = config.get("metricas_validacion", {}), config.get("metricas_test", {})
    return {"acc_validacion": val.get("accuracy", 0.978), "acc_test": test.get("accuracy", 0.867),
            "sensibilidad_test": test.get("recall_sensibilidad", 0.995),
            "especificidad_test": test.get("especificidad", 0.654)}


def construir_prompt(diagnostico: str, prob_neumonia: float, pregunta: str, metricas: Dict[str, float]) -> str:
    resultado = "signos compatibles con neumonía" if diagnostico == "PNEUMONIA" else "sin signos compatibles con neumonía"
    return (
        f"Resultado del modelo para esta radiografía: {resultado} (probabilidad de neumonía: {_pct(prob_neumonia)}).\n"
        f"Datos del modelo: exactitud de {_pct(metricas['acc_validacion'])} en validación y de "
        f"{_pct(metricas['acc_test'])} en un conjunto de prueba de otra población; en esa prueba detectó el "
        f"{_pct(metricas['sensibilidad_test'])} de las neumonías (sensibilidad), pero solo reconoció como normales "
        f"el {_pct(metricas['especificidad_test'])} de las radiografías sanas (especificidad), así que puede dar "
        f"falsos positivos.\n\n"
        f"Pregunta del usuario (responde solo a esto, siguiendo tus reglas): \"{pregunta}\""
    )


def respuesta_respaldo(clave: Optional[str], diagnostico: str, prob_neumonia: float,
                       metricas: Dict[str, float]) -> str:
    """Respuestas escritas a mano para cuando el LLM no está disponible o su texto no pasa el filtro."""
    if clave == "que_es":
        return ("La neumonía es una infección que inflama los alvéolos de uno o ambos pulmones, que pueden llenarse "
                "de líquido o pus. Puede deberse a virus, bacterias u hongos y suele causar tos, fiebre y dificultad "
                "para respirar. Su gravedad varía según la edad y el estado de salud de cada persona. "
                "Un médico es quien puede confirmarla y decidir el tratamiento.")
    if clave == "urgencias":
        return ("Acuda a urgencias si tiene dificultad para respirar, respiración muy rápida, labios o uñas azulados, "
                "dolor en el pecho, fiebre muy alta o que no baja, somnolencia o confusión. En niños, también son "
                "señales de alarma el hundimiento de las costillas al respirar, el quejido y el rechazo a comer o "
                "beber. Ante cualquiera de estas señales no espere: busque atención médica de inmediato.")
    if clave == "contagio":
        return ("La neumonía se contagia sobre todo por las gotitas que se expulsan al toser o estornudar y por "
                "tocarse la cara con las manos contaminadas. Para prevenirla, lávese las manos con frecuencia, "
                "cúbrase al toser, evite el humo de tabaco y mantenga al día las vacunas, como influenza y neumococo. "
                "Su médico puede indicarle qué vacunas le corresponden.")
    if clave == "confiable":
        return (f"El modelo acertó el {_pct(metricas['acc_validacion'])} de las veces en validación y el "
                f"{_pct(metricas['acc_test'])} en una prueba con radiografías de otra población. En esa prueba "
                f"detectó el {_pct(metricas['sensibilidad_test'])} de las neumonías, pero también marcó como "
                f"neumonía varias radiografías sanas, por lo que puede dar falsos positivos. Es un resultado "
                f"experimental y orientativo: solo un médico puede confirmarlo.")
    if clave == "que_hacer" and diagnostico == "PNEUMONIA":
        return (f"El modelo encontró signos compatibles con neumonía (probabilidad de {_pct(prob_neumonia)}), pero "
                "esto no es un diagnóstico. Lo recomendable es consultar a un médico lo antes posible para que lo "
                "confirme con un examen clínico. No se automedique y acuda a urgencias si tiene dificultad para "
                "respirar, dolor en el pecho o fiebre muy alta.")
    if clave == "que_hacer":
        return (f"El modelo no encontró signos compatibles con neumonía (probabilidad de {_pct(prob_neumonia)}), "
                "aunque esto no descarta otras enfermedades. Si tiene tos persistente, fiebre, dolor en el pecho o "
                "dificultad para respirar, consulte a un médico. Si no tiene síntomas, mantenga sus hábitos de "
                "prevención y sus controles de salud habituales.")
    return ("En este momento no puedo responder esa pregunta con seguridad. Puedo ayudarle con dudas sobre la "
            "neumonía, las señales de alarma, la prevención o la confiabilidad de este resultado. "
            "Para cualquier decisión sobre su salud, consulte a un médico.")


def _limpiar(texto: str) -> str:
    texto = texto.replace("**", "").replace("*", "").strip()
    texto = re.sub(r"\s+", " ", texto)
    oraciones = [o for o in re.split(r"(?<=[.!?])\s+", texto) if o]
    if oraciones and not oraciones[-1].endswith((".", "!", "?")):
        oraciones = oraciones[:-1]   # frase cortada a medias
    if len(oraciones) > MAX_ORACIONES:
        oraciones = oraciones[:MAX_ORACIONES]
    texto = " ".join(oraciones)
    if texto and not _MEDICO.search(texto):
        # Siempre debe sugerir consultar a un médico: si no lo hizo, se reemplaza la última oración.
        oraciones = oraciones[:MAX_ORACIONES - 1]
        texto = " ".join(oraciones + ["Consulte a un médico para valorar su caso."])
    return texto


def _motivo_rechazo(texto: str) -> Optional[str]:
    if len(texto) < MIN_CARACTERES:
        return f"respuesta corta ({len(texto)} caracteres)"
    for patron, motivo in ((_FARMACOS, "menciona fármacos o dosis"), (_PROMESAS, "promesa de cura"),
                           (_DIAGNOSTICA, "da un diagnóstico")):
        encontrado = patron.search(texto)
        if encontrado:
            return f"{motivo}: '{encontrado.group(0)}'"
    return None


def responder(diagnostico: str, prob_neumonia: float, pregunta: str, metricas: Dict[str, float]) -> Dict[str, object]:
    pregunta = re.sub(r"\s+", " ", pregunta).strip()
    clave = _CLAVE_POR_PREGUNTA.get(_normalizar(pregunta))
    texto: Optional[str] = None
    modelo_usado: Optional[str] = None
    motivo: Optional[str] = None
    if not llm.llm_cargado():
        motivo = llm._motivo_sin_llm or "LLM no disponible"
    else:
        try:
            crudo, modelo_usado = llm._llamar_llm(construir_prompt(diagnostico, prob_neumonia, pregunta, metricas),
                                                  INSTRUCCIONES_ASISTENTE)
            texto = _limpiar(crudo or "")
            motivo = "respuesta vacía del LLM" if not texto else _motivo_rechazo(texto)
        except Exception as exc:   # sin red, cuota agotada (429), timeout...
            motivo = f"error del LLM: {type(exc).__name__}: {str(exc)[:200]}"
        if motivo:
            print(f"Asistente: respuesta de respaldo -> {motivo}")
            texto = None
    generado = texto is not None
    return {
        "pregunta": pregunta,
        "respuesta": texto if generado else respuesta_respaldo(clave, diagnostico, prob_neumonia, metricas),
        "generado_por_llm": generado,
        "motivo_respaldo": motivo,
        "modelo_llm": modelo_usado if generado else None,
        "aviso": AVISO,
    }
