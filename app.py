

# -*- coding: utf-8 -*-
import os
import re
from typing import Dict, Any, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline
import whisper

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Conectividad SIG – Dashboard", layout="wide", page_icon="📊")
st.title("📊 Conectividad y Habilidades Digitales – Dashboard")
st.caption("Flujo 100% automático: KoboToolbox → Transcripción y análisis → Visualización")

# =========================
# SECRETS (TOKEN / FORM_ID)
# =========================
try:
    API_TOKEN = st.secrets["API_TOKEN"]
    FORM_ID = st.secrets["FORM_ID"]
except Exception:
    st.error("❌ Faltan secretos `API_TOKEN` y `FORM_ID` en Settings → Secrets (Streamlit Cloud).")
    st.stop()

BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets/"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# Campos de audio del XLSForm (ids)
AUDIO_FIELDS = {
    "q_conectividad_acceso": "Conectividad y calidad de internet",
    "q_uso_oportunidades": "Uso y oportunidades digitales",
    "q_derechos_riesgos": "Derechos y riesgos",
    "q_participacion_futuro": "Participación y futuro",
}

# =========================
# UTILIDADES
# =========================
def parse_geopoint(geo: str) -> Tuple[Optional[float], Optional[float]]:
    """Geopoint Kobo: 'lat lon alt acc' -> (lat, lon)"""
    if not isinstance(geo, str) or not geo.strip():
        return None, None
    parts = geo.split()
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def fetch_kobo_json_raw() -> List[Dict[str, Any]]:
    """Devuelve la lista 'results' cruda desde la API JSON."""
    url = f"{BASE_URL}{FORM_ID}/data/?format=json"
    st.write(f"**Debug:** GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    st.write(f"**Debug:** Código de respuesta: {r.status_code}")
    if r.status_code != 200:
        st.error(f"❌ Error {r.status_code} al obtener datos JSON desde Kobo.")
        return []
    payload = r.json()
    return payload.get("results", [])

@st.cache_data(show_spinner=False)
def normalize_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Normaliza JSON a DataFrame (todas las claves flaten)."""
    if not results:
        return pd.DataFrame()
    return pd.json_normalize(results)

def get_value_any(rec: Dict[str, Any], keys: List[str], default: str = "") -> Any:
    """Obtiene el primer valor disponible de una lista de claves."""
    for k in keys:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return default

def find_audio_url_for_field(rec: Dict[str, Any], field_name: str) -> Optional[str]:
    """
    Estrategia robusta:
    1) Si el JSON trae la llave '{field_name}_URL' o '{field_name}_url' -> usarla.
    2) Si no, buscar en rec['_attachments'] la coincidencia por 'field_name'.
    """
    # 1) Intento con claves _URL
    for candidate in [f"{field_name}_URL", f"{field_name}_url"]:
        url_val = rec.get(candidate)
        if isinstance(url_val, str) and url_val.startswith("http"):
            return url_val

    # 2) Buscar en adjuntos
    atts = rec.get("_attachments", [])
    for att in atts:
        att_field = att.get("field") or att.get("field_name")
        if att_field == field_name:
            dl = att.get("download_url", "")
            if dl:
                return f"https://eu.kobotoolbox.org{dl}"
    return None

def download_audio_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    # Modelo multilingüe: positivo/neutral/negativo
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def transcribe_audio(audio_bytes: Optional[bytes], whisper_model) -> str:
    if not audio_bytes:
        return ""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # Fijamos español; si quieres autodetección, elimina 'language'
        result = whisper_model.transcribe(tmp_path, language="es")
        text = (result.get("text") or "").strip()
    except Exception:
        text = ""
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return text

def map_sentiment(label: str) -> str:
    lb = (label or "").lower()
    if "neg" in lb: return "Negativo"
    if "pos" in lb: return "Positivo"
    if "neu" in lb: return "Neutro"
    return "Sin texto"

STOPWORDS = set("""
a al algo algunas algunos ante antes aquella aquellas aquello aquellos aquí así aun aunque bajo bien cada casi como con contra cuál cuales cuando de del desde donde dos el ella ellas ello ellos en entre era erais eran eras eres es esa esas ese esos esta estaban estáis estamos están estar esté esto estos fue fueron ha haber había habéis había habían hago hasta hay la las le les lo los más me mí mía mías mío míos mientras muy nos nosotras nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros para pero poca pocas poco pocos podrá porque por qué que quien quienes se sea según ser si sido sin sobre sois somos son su sus también te tendrá tiene todas toda todos todo tras tú tus un una unas unos y ya yo
""".split())

def extract_keywords(text: str, top_n: int = 10) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = re.sub(r"[^\wáéíóúñü ]+", " ", t, flags=re.UNICODE)
    tokens = [tok for tok in t.split() if len(tok) >= 4 and tok not in STOPWORDS]
    from collections import Counter
    freq = Counter(tokens)
    return ", ".join([w for w, _ in freq.most_common(top_n)])

# =========================
# 1) DESCARGA Y PREPARACIÓN
# =========================
st.subheader("1) Descargando datos de KoboToolbox")
results = fetch_kobo_json_raw()
st.write(f"**Debug:** Envios descargados: {len(results)}")

if not results:
    st.warning("No se encontraron envíos. Verifica el FORM_ID, permisos del token o que existan respuestas.")
    st.stop()

df_norm = normalize_results(results)
st.write(f"**Debug:** Columnas disponibles en JSON (muestra): {df_norm.columns.tolist()[:15]}")

# Construimos DF largo por PREGUNTA (una fila = un audio potencial)
rows: List[Dict[str, Any]] = []
for rec in results:
    # Campos de contexto (aceptamos tanto label como name)
    municipio = get_value_any(rec, ["municipio", "Municipio"], "")
    barrio    = get_value_any(rec, ["barrio_vereda", "Barrio/Vereda"], "")
    # Geopoint: raw 'ubicacion' o derivados
    lat, lon = None, None
    geo = rec.get("ubicacion")
    if isinstance(geo, str):
        lat, lon = parse_geopoint(geo)
    else:
        # Algunos exports traen columnas separadas
        lat = rec.get("_Por favor captura tu ubicación (GPS)_latitude")
        lon = rec.get("_Por favor captura tu ubicación (GPS)_longitude")

    for field_name, pregunta in AUDIO_FIELDS.items():
        audio_url = find_audio_url_for_field(rec, field_name)  # URL robusta (_URL o _attachments)
        rows.append({
            "pregunta": pregunta,
            "field_name": field_name,
            "audio_url": audio_url or "",
            "municipio": municipio,
            "barrio_vereda": barrio,
            "lat": lat,
            "lon": lon,
            "transcripcion": "",
            "sentimiento": "",
            "palabras_clave": "",
        })

df = pd.DataFrame(rows)
st.write(f"**Debug:** Filas construidas (4 por envío): {len(df)}")

# Filtrar audios válidos
df = df[df["audio_url"].astype(str).str.startswith("http")].reset_index(drop=True)
st.write(f"**Debug:** Filas con audio_url válido: {len(df)}")

if df.empty:
    st.warning("No hay audios válidos para procesar. Revisa que los envíos tengan adjuntos y que el token tenga acceso.")
    st.stop()

# =========================
# 2) TRANSCRIPCIÓN Y ANÁLISIS
# =========================
st.sidebar.header("Parámetros de procesamiento")
model_size = st.sidebar.selectbox("Tamaño modelo Whisper", ["tiny", "base", "small"], index=1,
                                  help="Modelos más grandes son más precisos pero más lentos.")
max_to_process = st.sidebar.slider("Máximo de audios a transcribir", 1, min(50, len(df)), min(10, len(df)),
                                   help="Útil para evitar timeouts en Streamlit Cloud.")

st.subheader("2) Transcribiendo audios y analizando texto")
try:
    whisper_model = load_whisper_model(model_size)
except Exception:
    st.error("❌ No se pudo cargar el modelo Whisper. Verifica dependencias (`torch`, `openai-whisper`) en requirements.txt")
    st.stop()

try:
    senti = load_sentiment_pipeline()
except Exception:
    st.error("❌ No se pudo cargar el modelo de sentimiento (transformers). Verifica dependencias.")
    st.stop()

progress = st.progress(0)
for i, idx in enumerate(df.index[:max_to_process], start=1):
    url = df.at[idx, "audio_url"]
    audio_bytes = download_audio_bytes(url)
    text = transcribe_audio(audio_bytes, whisper_model) if audio_bytes else ""
    df.at[idx, "transcripcion"] = text

    if text.strip():
        try:
            res = senti(text)[0]  # {'label': 'positive/neutral/negative', 'score': ...}
            df.at[idx, "sentimiento"] = map_sentiment(res.get("label"))
        except Exception:
            df.at[idx, "sentimiento"] = "Sin texto"
    else:
        df.at[idx, "sentimiento"] = "Sin texto"

    df.at[idx, "palabras_clave"] = extract_keywords(text)
    progress.progress(int(i / max_to_process * 100))
progress.empty()

n_trans = (df["transcripcion"].astype(str).str.strip() != "").sum()
st.write(f"**Debug:** Transcripciones generadas: {n_trans} / {len(df)}")

# =========================
# 3) DASHBOARD
# =========================
st.success("✅ ¡Procesamiento completo!")

# Filtros
st.sidebar.header("Filtros")
preguntas_sel = st.sidebar.multiselect("Pregunta", sorted(df["pregunta"].dropna().unique().tolist()))
municipios_sel = st.sidebar.multiselect("Municipio", sorted(df["municipio"].dropna().unique().tolist()))
sentimientos_sel = st.sidebar.multiselect("Sentimiento", ["Negativo", "Neutro", "Positivo", "Sin texto"])

df_f = df.copy()
if preguntas_sel:
    df_f = df_f[df_f["pregunta"].isin(preguntas_sel)]
if municipios_sel:
    df_f = df_f[df_f["municipio"].isin(municipios_sel)]
if sentimientos_sel:
    df_f = df_f[df_f["sentimiento"].isin(sentimientos_sel)]

# Distribución de sentimientos
st.subheader("Distribución de sentimientos")
fig, ax = plt.subplots(figsize=(7,4))
order = ["Negativo", "Neutro", "Positivo", "Sin texto"]
sns.countplot(data=df_f, x="sentimiento", order=order, palette="coolwarm", ax=ax)
ax.set_xlabel("")
ax.set_ylabel("Conteo")
st.pyplot(fig)

# Nube de palabras
st.subheader("Nube de palabras (transcripciones)")
texto = " ".join(df_f["transcripcion"].dropna().astype(str))
if texto.strip():
    wc = WordCloud(width=1200, height=500, background_color="white").generate(texto)
    fig_wc, ax_wc = plt.subplots(figsize=(12,6))
    ax_wc.imshow(wc, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)
else:
    st.info("No hay texto para generar la nube.")

# Tabla detallada
st.subheader("Detalle por respuesta")
st.dataframe(df_f[[
    "pregunta", "municipio", "barrio_vereda",
    "audio_url", "palabras_clave", "sentimiento", "transcripcion"
]])

# Reproductor de audio (muestra)
st.subheader("Escuchar audios (muestra)")
N = st.slider("Cantidad de audios a mostrar", 1, min(20, len(df_f)), min(10, len(df_f)))
for _, row in df_f.head(N).iterrows():
    st.markdown(f"**{row['pregunta']} – {row['municipio']}**")
    b = download_audio_bytes(row["audio_url"])
    if b:
        st.audio(b, format="audio/mp3")
    else:
        st.info("Audio no disponible.")
    st.caption(f"Palabras clave: {row['palabras_clave']}")
    if row["transcripcion"]:
        with st.expander("Ver transcripción"):
            st.write(row["transcripcion"])
    st.divider()

# Mapa
st.subheader("Mapa de envíos")
map_df = df_f.dropna(subset=["lat", "lon"])[["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"})
if not map_df.empty:
    st.map(map_df, zoom=10)
else:
    st.info("No hay coordenadas GPS en los envíos seleccionados.")

# Exportación
st.download_button(
    label="Descargar CSV procesado",
    data=df_f.to_csv(index=False),
    file_name="resultados_encuesta.csv",
    mime="text/csv",
)

