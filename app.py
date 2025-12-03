
import os
import re
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
    FORM_ID = st.secrets["FORM_ID"]  # Debe ser: a6XQZtj52WmfUix3KCDdpV
except Exception:
    st.error("❌ Faltan secretos API_TOKEN y FORM_ID en Settings → Secrets")
    st.stop()

BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets/"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# =========================
# FUNCIONES
# =========================
def parse_geopoint(geo: str):
    if not isinstance(geo, str): return None, None
    parts = geo.split()
    if len(parts) >= 2:
        try: return float(parts[0]), float(parts[1])
        except: return None, None
    return None, None

@st.cache_data(show_spinner=False)
def fetch_kobo_json():
    url = f"{BASE_URL}{FORM_ID}/data/?format=json"
    st.write(f"**Debug:** GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    st.write(f"**Debug:** Código de respuesta: {r.status_code}")
    if r.status_code != 200:
        st.error(f"❌ Error {r.status_code} al obtener datos JSON")
        return []
    return r.json().get("results", [])

def find_audio_url(rec: dict, field_name: str):
    # 1) columnas *_URL
    for k in (f"{field_name}_URL", f"{field_name}_url"):
        v = rec.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    # 2) adjuntos
    for att in rec.get("_attachments", []):
        if (att.get("field") or att.get("field_name")) == field_name:
            dl = att.get("download_url", "")
            if dl:
                return f"https://eu.kobotoolbox.org{dl}"
    return None

def download_audio_bytes(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 200: return r.content
    except: return None
    return None

@st.cache_resource
def load_whisper(model_size="base"):
    return whisper.load_model(model_size)

@st.cache_resource
def load_senti():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def transcribe(audio_bytes, model):
    if not audio_bytes: return ""
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes); path = tmp.name
    try:
        res = model.transcribe(path, language="es")
        txt = (res.get("text") or "").strip()
    except: txt = ""
    finally:
        try: os.remove(path)
        except: pass
    return txt

STOPWORDS = set("""
a al algo algunas algunos ante antes aquella aquellas aquello aquellos aquí así aun aunque bajo bien cada casi como con contra cuál cuales cuando de del desde donde dos el ella ellas ello ellos en entre era erais eran eras eres es esa esas ese esos esta estaban estáis estamos están estar esté esto estos fue fueron ha haber había habéis había habían hago hasta hay la las le les lo los más me mí mía mías mío míos mientras muy nos nosotras nosotros nuestra nuestras nuestro nuestros o os otra otras otro otros para pero poca pocas poco pocos podrá porque por qué que quien quienes se sea según ser si sido sin sobre sois somos son su sus también te tendrá tiene todas toda todos todo tras tú tus un una unas unos y ya yo
""".split())

def keywords(text, top=10):
    if not isinstance(text, str) or not text.strip(): return ""
    t = re.sub(r"[^\wáéíóúñü ]+", " ", text.lower())
    toks = [w for w in t.split() if len(w) >= 4 and w not in STOPWORDS]
    from collections import Counter
    return ", ".join([w for w,_ in Counter(toks).most_common(top)])

# =========================
# 1) DESCARGA Y PREPARACIÓN
# =========================
st.subheader("1) Descargando datos de KoboToolbox")
results = fetch_kobo_json()
st.write(f"**Debug:** Envios descargados: {len(results)}")
if not results:
    st.warning("No hay envíos o no se pudo cargar datos.")
    st.stop()

# Construcción DF largo
AUDIO_FIELDS = {
    "q_conectividad_acceso": "Conectividad y calidad de internet",
    "q_uso_oportunidades": "Uso y oportunidades digitales",
    "q_derechos_riesgos": "Derechos y riesgos",
    "q_participacion_futuro": "Participación y futuro",
}
rows = []
for rec in results:
    municipio = rec.get("municipio") or rec.get("Municipio") or ""
    barrio    = rec.get("barrio_vereda") or rec.get("Barrio/Vereda") or ""
    lat, lon  = None, None
    if isinstance(rec.get("ubicacion"), str):
        lat, lon = parse_geopoint(rec["ubicacion"])
    else:
        lat = rec.get("_Por favor captura tu ubicación (GPS)_latitude")
        lon = rec.get("_Por favor captura tu ubicación (GPS)_longitude")
    for field_name, pregunta in AUDIO_FIELDS.items():
        url = find_audio_url(rec, field_name)
        rows.append({
            "pregunta": pregunta,
            "field_name": field_name,
            "audio_url": url or "",
            "municipio": municipio,
            "barrio_vereda": barrio,
            "lat": lat, "lon": lon,
            "transcripcion": "", "sentimiento": "", "palabras_clave": "",
        })

df = pd.DataFrame(rows)
st.write(f"**Debug:** Filas construidas: {len(df)}")
df = df[df["audio_url"].astype(str).str.startswith("http")].reset_index(drop=True)
st.write(f"**Debug:** Filas con audio_url válido: {len(df)}")
if df.empty:
    st.warning("No se detectaron URLs de audio.")
    st.stop()

# =========================
# 2) TRANSCRIPCIÓN Y ANÁLISIS
# =========================
st.sidebar.header("Parámetros de procesamiento")
model_size = st.sidebar.selectbox("Modelo Whisper", ["tiny","base","small"], index=1)
max_to_process = st.sidebar.slider("Máximo de audios a transcribir", 1, min(50, len(df)), min(10, len(df)))

st.subheader("2) Transcribiendo audios y analizando")
wh_model = load_whisper(model_size)
senti    = load_senti()
prog = st.progress(0)

for i, idx in enumerate(df.index[:max_to_process], start=1):
    b   = download_audio_bytes(df.at[idx, "audio_url"])
    txt = transcribe(b, wh_model)
    df.at[idx, "transcripcion"] = txt
    if txt.strip():
        try:
            out = senti(txt)[0]
            lab = out.get("label","").lower()
            df.at[idx, "sentimiento"] = "Positivo" if "pos" in lab else "Negativo" if "neg" in lab else "Neutro"
        except:
            df.at[idx, "sentimiento"] = "Sin texto"
    else:
        df.at[idx, "sentimiento"] = "Sin texto"
    df.at[idx, "palabras_clave"] = keywords(txt)
    prog.progress(int(i / max_to_process * 100))
prog.empty()

st.write(f"**Debug:** Transcripciones generadas: {(df['transcripcion'].astype(str).str.strip()!='').sum()} / {len(df)}")

# =========================
# 3) DASHBOARD
# =========================
st.success("✅ ¡Procesamiento completo!")
st.sidebar.header("Filtros")
preg_sel = st.sidebar.multiselect("Pregunta", sorted(df["pregunta"].dropna().unique().tolist()))
mun_sel  = st.sidebar.multiselect("Municipio", sorted(df["municipio"].dropna().unique().tolist()))
sent_sel = st.sidebar.multiselect("Sentimiento", ["Negativo","Neutro","Positivo","Sin texto"])

df_f = df.copy()
if preg_sel: df_f = df_f[df_f["pregunta"].isin(preg_sel)]
if mun_sel:  df_f = df_f[df_f["municipio"].isin(mun_sel)]
if sent_sel: df_f = df_f[df_f["sentimiento"].isin(sent_sel)]

st.subheader("Distribución de sentimientos")
fig, ax = plt.subplots(figsize=(7,4))
sns.countplot(data=df_f, x="sentimiento", order=["Negativo","Neutro","Positivo","Sin texto"], palette="coolwarm", ax=ax)
ax.set_xlabel(""); ax.set_ylabel("Conteo")
st.pyplot(fig)

st.subheader("Nube de palabras (transcripciones)")
texto = " ".join(df_f["transcripcion"].dropna().astype(str))
if texto.strip():
    wc = WordCloud(width=1200, height=500, background_color="white").generate(texto)
    fig_wc, ax_wc = plt.subplots(figsize=(12,6))
    ax_wc.imshow(wc, interpolation="bilinear"); ax_wc.axis("off")
    st.pyplot(fig_wc)
else:
    st.info("No hay texto para generar la nube.")

st.subheader("Detalle por respuesta")
st.dataframe(df_f[["pregunta","municipio","barrio_vereda","audio_url","palabras_clave","sentimiento","transcripcion"]])

st.subheader("Escuchar audios (muestra)")
N = st.slider("Cantidad de audios a mostrar", 1, min(20, len(df_f)), min(10, len(df_f)))
for _, row in df_f.head(N).iterrows():
    st.markdown(f"**{row['pregunta']} – {row['municipio']}**")
    b = download_audio_bytes(row["audio_url"])
    if b: st.audio(b, format="audio/mp3")
    else: st.info("Audio no disponible.")
    st.caption(f"Palabras clave: {row['palabras_clave']}")
    if row["transcripcion"]:
        with st.expander("Ver transcripción"): st.write(row["transcripcion"])
    st.divider()

st.subheader("Mapa de envíos")
map_df = df_f.dropna(subset=["lat","lon"])[["lat","lon"]].rename(columns={"lat":"latitude","lon":"longitude"})
if not map_df.empty: st.map(map_df, zoom=10)
else: st.info("No hay coordenadas GPS en los envíos seleccionados.")

st.download_button("Descargar CSV procesado", df_f.to_csv(index=False), "resultados_encuesta.csv", "text/csv")

