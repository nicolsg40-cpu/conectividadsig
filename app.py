
import os
import re
import time
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
# TOKEN Y FORM_ID DESDE SECRETS
# =========================
try:
    API_TOKEN = st.secrets["API_TOKEN"]
    FORM_ID = st.secrets["FORM_ID"]
except Exception:
    st.error("❌ Faltan secretos API_TOKEN y FORM_ID en Settings → Secrets de Streamlit Cloud.")
    st.stop()

BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets/"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# =========================
# FUNCIONES
# =========================
@st.cache_data(show_spinner=False)
def fetch_kobo_csv() -> pd.DataFrame:
    # Intento 1: endpoint directo con ?format=csv
    url_direct = f"{BASE_URL}{FORM_ID}/data/?format=csv"
    st.write(f"**Debug:** Intentando descargar CSV desde {url_direct}")
    r = requests.get(url_direct, headers=HEADERS, timeout=60)
    st.write(f"**Debug:** Código de respuesta: {r.status_code}")
    if r.status_code == 200:
        from io import StringIO
        return pd.read_csv(StringIO(r.text))

    # Intento 2: usar exportación
    st.warning("Intento directo falló. Creando exportación...")
    export_url = f"{BASE_URL}{FORM_ID}/exports/"
    payload = {"format": "csv"}
    resp = requests.post(export_url, headers=HEADERS, json=payload)
    if resp.status_code != 201:
        st.error(f"❌ Error {resp.status_code} al crear exportación.")
        return pd.DataFrame()
    export_info = resp.json()
    export_download_url = export_info.get("result", {}).get("download_url")
    if not export_download_url:
        st.error("❌ No se obtuvo URL de descarga en la exportación.")
        return pd.DataFrame()
    # Descargar el archivo exportado
    full_download_url = f"https://eu.kobotoolbox.org{export_download_url}"
    st.write(f"**Debug:** Descargando exportación desde {full_download_url}")
    time.sleep(5)  # esperar a que se genere el archivo
    r2 = requests.get(full_download_url, headers=HEADERS, timeout=60)
    if r2.status_code != 200:
        st.error(f"❌ Error {r2.status_code} al descargar exportación.")
        return pd.DataFrame()
    from io import StringIO
    return pd.read_csv(StringIO(r2.text))

def download_audio_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            return r.content
    except Exception:
        return None
    return None

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def transcribe_audio(audio_bytes: bytes, whisper_model) -> str:
    if not audio_bytes:
        return ""
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
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
# EJECUCIÓN
# =========================
st.subheader("1) Descargando datos de KoboToolbox")
df_raw = fetch_kobo_csv()
st.write(f"**Debug:** Registros descargados: {len(df_raw)}")

if df_raw.empty:
    st.warning("No se pudo cargar datos desde Kobo.")
    st.stop()

# Detectar columnas *_URL
url_cols = [c for c in df_raw.columns if c.endswith("_URL")]
st.write(f"**Debug:** Columnas de audio detectadas: {url_cols}")

if not url_cols:
    st.warning("No se detectaron columnas *_URL en el CSV. Revisa la exportación.")
    st.stop()

# Construir DF largo
rows = []
for i, rec in df_raw.iterrows():
    municipio = rec.get("Municipio", "")
    barrio = rec.get("Barrio/Vereda", "")
    lat = rec.get("_Por favor captura tu ubicación (GPS)_latitude")
    lon = rec.get("_Por favor captura tu ubicación (GPS)_longitude")
    for col in url_cols:
        pregunta = col.replace("_URL", "")
        audio_url = rec.get(col)
        rows.append({
            "pregunta": pregunta,
            "municipio": municipio,
            "barrio_vereda": barrio,
            "audio_url": audio_url,
            "lat": lat,
            "lon": lon,
            "transcripcion": "",
            "sentimiento": "",
            "palabras_clave": ""
        })

df = pd.DataFrame(rows)
st.write(f"**Debug:** Filas construidas: {len(df)}")

# Filtrar audios válidos
df = df[df["audio_url"].astype(str).str.startswith("http")].reset_index(drop=True)
st.write(f"**Debug:** Filas con audio_url válido: {len(df)}")

if df.empty:
    st.warning("No hay audios válidos para procesar.")
    st.stop()

# Parámetros
st.sidebar.header("Parámetros de procesamiento")
model_size = st.sidebar.selectbox("Tamaño modelo Whisper", ["tiny", "base", "small"], index=1)
max_to_process = st.sidebar.slider("Máximo de audios a transcribir", 1, min(50, len(df)), min(10, len(df)))

# Procesamiento
st.subheader("2) Transcribiendo audios y analizando texto")
whisper_model = load_whisper_model(model_size)
senti = load_sentiment_pipeline()

progress = st.progress(0)
processed = 0
for idx in df.index[:max_to_process]:
    url = df.at[idx, "audio_url"]
    b = download_audio_bytes(url)
    text = transcribe_audio(b, whisper_model) if b else ""
    df.at[idx, "transcripcion"] = text
    if text.strip():
        try:
            res = senti(text)[0]
            df.at[idx, "sentimiento"] = map_sentiment(res.get("label", ""))
        except Exception:
            df.at[idx, "sentimiento"] = "Sin texto"
    else:
        df.at[idx, "sentimiento"] = "Sin texto"
    df.at[idx, "palabras_clave"] = extract_keywords(text)
    processed += 1
    progress.progress(int(processed / max_to_process * 100))
progress.empty()

st.write(f"**Debug:** Transcripciones generadas: {(df['transcripcion'].astype(str).str.strip()!='').sum()} / {len(df)}")

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
st.dataframe(df_f[["pregunta","municipio","audio_url","palabras_clave","sentimiento","transcripcion"]])

# Reproductor de audio
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
map_df = df_f.dropna(subset=["lat","lon"])[["lat","lon"]].rename(columns={"lat":"latitude","lon":"longitude"})
if not map_df.empty:
    st.map(map_df, zoom=10)
else:
    st.info("No hay coordenadas GPS en los envíos seleccionados.")

# Exportación
st.download_button(label="Descargar CSV procesado", data=df_f.to_csv(index=False), file_name="resultados_encuesta.csv", mime="text/csv")
