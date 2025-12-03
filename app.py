
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
    url = f"{BASE_URL}{FORM_ID}/data.csv"
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        st.error(f"❌ Error {r.status_code} al obtener CSV desde Kobo.")
        return pd.DataFrame()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

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

