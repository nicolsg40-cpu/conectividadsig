import os
import re
import time
from io import BytesIO, StringIO
from typing import Dict, Any, List, Optional

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
    st.error("Faltan secretos API_TOKEN y FORM_ID en la configuración de Streamlit Cloud.")
    st.stop()

BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets/"
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# =========================
# FUNCIONES
# =========================
@st.cache_data(show_spinner=False)
def fetch_kobo_json() -> List[Dict[str, Any]]:
    url = f"{BASE_URL}{FORM_ID}/data/?format=json"
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        st.error(f"Error {r.status_code} al obtener datos desde Kobo.")
        return []
    payload = r.json()
    return payload.get("results", [])

# =========================
# BLOQUE DE DEBUG
# =========================
st.subheader("1) Descargando datos de KoboToolbox")
records = fetch_kobo_json()
st.write(f"Debug: Cantidad de registros descargados: {len(records)}")

if not records:
    st.warning("No se encontraron registros. Verifica que el formulario tenga envíos y audios.")
    st.stop()

# Aquí continuarías con el flujo completo: preparar DataFrame, transcribir audios, análisis de sentimiento, palabras clave, dashboard.
