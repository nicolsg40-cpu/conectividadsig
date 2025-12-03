
# -*- coding: utf-8 -*-
import os
import re
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Modelos
from transformers import pipeline
import whisper

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="Conectividad SIG – Dashboard", layout="wide", page_icon="📊")
st.title("📊 Conectividad y Habilidades Digitales – Dashboard")
st.caption("Flujo 100% automático: KoboToolbox → Transcripción y análisis → Visualización")

# =========================
# TOKEN Y FORM_ID DESDE SECRETS
# =========================
try:
    API_TOKEN = st.secrets["API_TOKEN"]
    FORM_ID = st.secrets["FORM_ID"]
except Exception:
    st.error("❌ Faltan secretos `API_TOKEN` y `FORM_ID` en Settings → Secrets de Streamlit Cloud.")
    st.stop()

BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets/"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# =========================
# CAMPOS DEL FORMULARIO
# =========================
AUDIO_FIELDS = {
    "q_conectividad_acceso": "Conectividad y calidad de internet",
    "q_uso_oportunidades": "Uso y oportunidades digitales",
    "q_derechos_riesgos": "Derechos y riesgos",
    "q_participacion_futuro": "Participación y futuro",
}
GEO_FIELD = "ubicacion"  # "lat lon alt acc"

# =========================
# UTILIDADES
# =========================
def parse_geopoint(geo: str) -> (Optional[float], Optional[float]):
    if not isinstance(geo, str) or not geo.strip():
        return None, None
    parts = geo.split()
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None, None

@st.cache_data(show_spinner=False)
def fetch_kobo_json() -> List[Dict[str, Any]]:
    """Descarga datos del formulario en formato JSON (incluye adjuntos)."""
    url = f"{BASE_URL}{FORM_ID}/data/?format=json"
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        st.error(f"❌ Error {r.status_code} al obtener datos desde Kobo.")
        return []
    payload = r.json()
    return payload.get("results", [])

def find_attachment_url(record: Dict[str, Any], filename_or_value: str) -> Optional[str]:
    """Busca el URL de descarga del adjunto asociado a un campo de audio."""
    if not filename_or_value:
        return None
    attachments = record.get("_attachments", [])
    for att in attachments:
        fname = att.get("filename", "")
        dl = att.get("download_url", "")
        if not dl:
            continue
        full_url = f"https://eu.kobotoolbox.org{dl}"
        if filename_or_value == fname or filename_or_value in full_url:
            return full_url
    return None

def download_audio_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            return r.content
        return None
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    """Carga el modelo Whisper (tiny|base|small) y lo cachea en la sesión."""
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Pipeline multilingüe (positivo/neutral/negativo)."""
    # cardiffnlp/twitter-xlm-roberta-base-sentiment
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def transcribe_audio(audio_bytes: bytes, whisper_model) -> str:
    if not audio_bytes:
        return ""
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        # Forzar idioma español; puedes remover language para auto-detec.
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
    label = (label or "").lower()
    if "neg" in label: return "Negativo"
    if "pos" in label: return "Positivo"
    if "neu" in label: return "Neutro"
    return "Sin texto"

SPANISH_STOPWORDS = set("""
a al algo algunas algunos ante antes aquella aquellas aquello aquellos aquí así aun aunque
bajo bien cada casi como con contra cuál cuales cuando de del desde donde dos el ella ellas
ello ellos en entre era erais eran eras eres es esa esas ese esos esta estaban estáis estamos
están estar esté esto estos fue fueron ha haber había habéis había habían hago hasta hay la las
le les lo los más me mí mía mías mío míos mientras muy nos nosotras nosotros nuestra nuestras
nuestro nuestros o os otra otras otro otros para pero poca pocas poco pocos podrá porque por
qué que quien quienes se sea según ser si sido sin sobre sois somos son su sus también te
tendrá tiene todas toda todos todo tras tú tus un una unas unos y ya yo
""".split())

def extract_keywords(text: str, top_n: int = 10) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = re.sub(r"[^\wáéíóúñü ]+", " ", t, flags=re.UNICODE)
    tokens = [tok for tok in t.split() if len(tok) >= 4 and tok not in SPANISH_STOPWORDS]
    from collections import Counter
    freq = Counter(tokens)
    return ", ".join([w for w, _ in freq.most_common(top_n)])

@st.cache_data(show_spinner=True)
def build_long_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        submission_id = rec.get("_id")
        municipio = rec.get("municipio", "")
        barrio = rec.get("barrio_vereda", "")
        lat, lon = parse_geopoint(rec.get(GEO_FIELD, ""))

        for field_name, pregunta in AUDIO_FIELDS.items():
            value = rec.get(field_name)  # suele ser el filename del adjunto
            audio_url = find_attachment_url(rec, value) if isinstance(value, str) else None
            rows.append({
                "submission_id": submission_id,
                "pregunta": pregunta,
                "field_name": field_name,
                "audio_filename": value if isinstance(value, str) else "",
                "audio_url": audio_url or "",
                "municipio": municipio,
                "barrio_vereda": barrio,
                "lat": lat,
                "lon": lon,
                "transcripcion": "",
                "sentimiento": "",
                "palabras_clave": "",
            })
    return pd.DataFrame(rows)

def process_audios_and_text(df: pd.DataFrame, model_size: str, max_to_process: int) -> pd.DataFrame:
    """Descarga, transcribe y analiza texto; respeta un máximo por corrida."""
    # Cargar modelos (cacheados)
    try:
        whisper_model = load_whisper_model(model_size)
    except Exception as e:
        st.error("❌ No se pudo cargar el modelo Whisper. Revisa dependencias (torch/whisper).")
        st.stop()
    try:
        senti = load_sentiment_pipeline()
    except Exception:
        st.error("❌ No se pudo cargar el modelo de sentimiento. Revisa dependencias de transformers.")
        st.stop()

    idxs = df.index.tolist()
    to_process = idxs[:max_to_process]  # limitar para evitar timeouts
    progress = st.progress(0)
    processed = 0

    for idx in idxs:
        audio_url = df.at[idx, "audio_url"]
        transcripcion = ""

        if idx in to_process and audio_url:
            b = download_audio_bytes(audio_url)
            transcripcion = transcribe_audio(b, whisper_model) if b else ""

        df.at[idx, "transcripcion"] = transcripcion

        # Sentimiento
        if transcripcion.strip():
            try:
                res = senti(transcripcion)[0]
                df.at[idx, "sentimiento"] = map_sentiment(res.get("label", ""))
            except Exception:
                df.at[idx, "sentimiento"] = "Sin texto"
        else:
            df.at[idx, "sentimiento"] = "Sin texto"

        # Palabras clave
        df.at[idx, "palabras_clave"] = extract_keywords(transcripcion)

        processed += 1
        progress.progress(min(int(processed / len(idxs) * 100), 100))

    progress.empty()
    return df

# =========================
# EJECUCIÓN
# =========================
st.subheader("1) Descargando datos de KoboToolbox")
records = fetch_kobo_json()
st.write(f"**Debug:** Cantidad de registros descargados: {len(records)}")

if not records:
    st.warning("No se encontraron registros. Verifica que el formulario tenga envíos y audios.")
    st.stop()

st.subheader("2) Preparando estructura de análisis")
df = build_long_dataframe(records)
st.write(f"**Debug:** Filas (una por pregunta): {len(df)}")

# Filtrar filas con URL de audio
df = df[df["audio_url"].astype(str).str.len() > 0].reset_index(drop=True)
st.write(f"**Debug:** Filas con audio_url válido: {len(df)}")

if df.empty:
    st.warning("No se detectaron adjuntos de audio. Revisa permisos del token y que existan audios en los envíos.")
    st.stop()

# Controles en la barra lateral
st.sidebar.header("Parámetros de procesamiento")
model_size = st.sidebar.selectbox("Tamaño modelo Whisper", options=["tiny", "base", "small"], index=1,
                                  help="Modelos más grandes = mejor calidad pero más lentos.")
max_to_process = st.sidebar.slider("Máximo de audios a transcribir en esta corrida",
                                   min_value=1, max_value=min(50, len(df)), value=min(20, len(df)),
                                   help="Útil para evitar timeouts en Streamlit Cloud.")

st.subheader("3) Transcribiendo audios y analizando texto")
df = process_audios_and_text(df, model_size=model_size, max_to_process=max_to_process)

# Métricas de debug
n_trans = (df["transcripcion"].astype(str).str.strip() != "").sum()
st.write(f"**Debug:** Transcripciones generadas: {n_trans} / {len(df)}")

# =========================
# DASHBOARD
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
sns.countplot(x=df_f["sentimiento"], order=order, palette="coolwarm", ax=ax)
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
    "audio_filename", "palabras_clave", "sentimiento", "transcripcion"
]])

# Reproductor de audio (muestra)
st.subheader("Escuchar audios (muestra)")
N = st.slider("Cantidad de audios a mostrar", min_value=1, max_value=min(20, len(df_f)), value=min(10, len(df_f)))
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
