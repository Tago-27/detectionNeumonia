# Detección de neumonía en radiografías de tórax + LLM + aprendizaje por refuerzo

API FastAPI con interfaz web que:

1. **Clasifica** radiografías de tórax en **NORMAL** o **PNEUMONIA** con una CNN (ensamble CNN propia + DenseNet121).
2. **Genera una recomendación** de más de 200 caracteres con un LLM (Google Gemini). La predicción es el input del prompt.
3. **Aprende con el feedback**: el usuario califica la recomendación con 👍 o 👎 y un agente de Q-learning aprende qué estilo de recomendación prefieren los usuarios.

Todo cabe en el **plan gratuito de Render** (512 MB): la API usa ~150 MB de RAM.

```text
imagen → POST /predecir → preprocesamiento → CNN (TFLite) → (diagnóstico, probabilidad)
       → agente RL elige el estilo → prompt con la predicción → LLM (Gemini) → filtro → JSON → web
usuario 👍/👎 → POST /rl/feedback → el agente actualiza su tabla Q → Postgres
```

## Estructura

```text
├── api/
│   ├── main.py                  ← servidor FastAPI (CNN + LLM + RL)
│   ├── preprocesamiento.py      ← mismo preprocesamiento que en el entrenamiento
│   ├── recomendaciones_llm.py   ← prompt, llamada a Gemini, filtro y texto de respaldo
│   └── agente_rl.py             ← agente Q-learning y almacenamiento de la tabla Q
├── modelo/
│   ├── modelo_neumonia_cnn_APROBADO_0.9769.tflite   ← modelo que usa la API (30 MB)
│   ├── config_modelo_APROBADO_0.9769.json           ← filtro, umbral y métricas
│   └── modelo_neumonia_cnn_APROBADO_0.9769.keras    ← original (solo local, no se sube: .gitignore)
├── herramientas/convertir_a_tflite.py   ← .keras → .tflite, con verificación
├── web/index.html               ← interfaz (HTML + CSS + JS) con botones de feedback
├── requirements.txt
├── render.yaml                  ← Blueprint de Render (plan free)
└── .python-version              ← Python 3.12
```

## Por qué TFLite

TensorFlow completo ocupa ~565 MB de RAM, más que los 512 MB de Render gratis. Por eso el modelo aprobado se convierte a `.tflite` y la API lo ejecuta con el intérprete ligero LiteRT (`ai-edge-litert`).

Se verificó con una radiografía real y variantes de ella que el `.tflite` da **las mismas probabilidades** que el `.keras`: diferencia máxima de 2,4·10⁻⁷.

Si reentrenas el modelo, pon el nuevo `.keras` y su `config*.json` en `modelo/` (borra los anteriores) y ejecuta:

```bash
pip install tensorflow==2.20.0 keras==3.15.1      # solo en local
python herramientas/convertir_a_tflite.py ruta/radiografia1.jpg ruta/radiografia2.jpg
```

## LLM: cómo se genera la recomendación

- **Input:** el diagnóstico y la probabilidad de neumonía que da la CNN, más una lista cerrada de consejos validados (`CONSEJOS`) y el estilo elegido por el agente RL.
- **Modelo:** Google Gemini, llamado por su API gratuita. Al arrancar, la API consulta los modelos disponibles para la clave y elige el Gemini Flash estable más reciente (Google renombra sus modelos a menudo). Se puede fijar uno con la variable `MODELO_LLM`.
- **Filtro:** se descarta el texto si tiene menos de 200 caracteres, menciona medicamentos o dosis, promete curas, saluda o hace preguntas. En ese caso, o si Gemini falla (sin clave, cuota agotada, timeout), se responde con un texto de respaldo armado con los consejos, también de más de 200 caracteres. El JSON indica cuál se usó con `generado_por_llm`.

## Aprendizaje por refuerzo

| Elemento   | Valor |
|------------|-------|
| Estado     | diagnóstico + confianza (`alta` ≥ 0,85, `media` ≥ 0,65, `baja`), p. ej. `PNEUMONIA\|alta` |
| Acciones   | estilos: `breve_directa`, `detallada`, `alarma_prevencion`, `empatica` |
| Recompensa | +1 me gusta, −1 no me gusta |
| Política   | ε-greedy (ε = 0,2, decae ×0,99 por feedback hasta 0,05) |
| Regla      | `Q(s,a) ← Q(s,a) + α·(r − Q(s,a))`, α = 0,2, γ = 0 (cada recomendación es un episodio de un paso) |

La tabla Q se guarda en **Postgres** si existe `DATABASE_URL`. Así no se pierde aunque Render apague el servicio. Sin esa variable (uso local) se guarda en `datos/q_table_recomendaciones.json`. Se puede consultar en `GET /rl/estado`.

## Endpoints

| Método | Ruta           | Descripción |
|--------|----------------|-------------|
| GET    | `/`            | Interfaz web |
| GET    | `/salud`       | Estado del servidor, el modelo, el LLM y la tabla Q |
| POST   | `/predecir`    | Recibe `imagen` (JPG/PNG, máx. 10 MB) y devuelve el diagnóstico, la recomendación y los datos del RL |
| POST   | `/rl/feedback` | `{"estado": "...", "accion": "...", "recompensa": 1}` |
| GET    | `/rl/estado`   | Tabla Q, ε y feedback acumulado |
| GET    | `/docs`        | Documentación interactiva |

## Despliegue en Render (gratis)

1. **Clave de Gemini:** créala gratis en <https://aistudio.google.com/apikey>.
2. **Base de datos:** crea un Postgres gratuito en [Neon](https://neon.tech) o [Supabase](https://supabase.com) y copia su *connection string*. Tiene la forma `postgresql://usuario:clave@host/base?sslmode=require`. La tabla se crea sola.
3. **GitHub:** sube el proyecto. El `.keras` queda fuera por `.gitignore` y el `.tflite` pesa 30 MB.
4. **Render:** entra en **New → Blueprint** y elige el repositorio. Te pedirá `GEMINI_API_KEY` y `DATABASE_URL`: pega los valores de los pasos 1 y 2.
5. Abre la URL del servicio y comprueba en `/salud` que diga `"llm_cargado": true` y `"tabla_q": "Postgres"`.

Limitaciones del plan gratuito:

- El servicio se apaga tras 15 minutos sin visitas y la siguiente visita tarda alrededor de un minuto en responder.
- La cuota gratuita de Gemini tiene un límite diario. Si se agota, se usa el texto de respaldo hasta el día siguiente.

## Ejecución local

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

Abre `http://127.0.0.1:8000`. Para usar Gemini en local, antes de `uvicorn` ejecuta en PowerShell `$env:GEMINI_API_KEY="tu-clave"`.

> El dataset solo contiene radiografías pediátricas, así que con radiografías de adultos el resultado es menos fiable.
> **Proyecto académico: no es una herramienta de diagnóstico médico.**
