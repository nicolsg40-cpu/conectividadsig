
# -*- coding: utf-8 -*-
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

# --- Modelos (cargados cuando se usan) ---
# Whisper para transcripción
# Transformers para sentimiento
from transformers import pipeline
import whisper

# =========================
# PARÁMETROS DE KOBO
# =========================
API_TOKEN = st.secrets["API_TOKEN"]
BASE_URL = "https://eu.kobotoolbox.org/api/v2/assets/"
FORM_ID = st.secrets["FORM_ID"]
HEADERS = {"Authorization": f"Token {API_TOKEN}"}

# Enlaces útiles:
# CSV directo: https://eu.kobotoolbox.org/api/v2/assets/{FORM_ID}/data.csv
# JSON con adjuntos: https://eu.kobotoolbox.org/api/v2/assets/{FORM_ID}/data/?format=json

# Campos del formulario (según tu XLSForm)
AUDIO_FIELDS = {
    "q_conectividad_acceso": "Conectividad y calidad de internet",
    "q_uso_oportunidades": "Uso y oportunidades digitales",
    "q_derechos_riesgos": "Derechos y riesgos",
    "q_participacion_futuro": "Participación y futuro",
}
TEXT_FIELDS = [
    "enumerator_name",       # Tu nombre (encuestador/a)
    "participant_code",      # Código participante (si aplica)
    "municipio",             # Municipio
    "barrio_vereda",         # Barrio/Vereda
]
GEO_FIELD = "ubicacion"      # geopoint: "lat lon alt acc"

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Conectividad SIG – Dashboard",
    layout="wide",
    page_icon="📊"
)
st.title("📊 Conectividad y Habilidades Digitales – Dashboard")
st.caption("Flujo 100% automático: KoboToolbox → Transcripción y análisis → Visualización")

# =========================
# UTILIDADES
# =========================
def parse_geopoint(geo: str) -> (Optional[float], Optional[float]):
    """Recibe 'lat lon alt acc' y retorna (lat, lon)."""
    if not isinstance(geo, str) or not geo.strip():
        return None, None
    parts = geo.split()
    try:
        lat = float(parts[0])
        lon = float(parts[1])
        return lat, lon
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def fetch_kobo_json() -> List[Dict[str, Any]]:
    """Descarga datos del formulario en formato JSON, incluyendo adjuntos."""
    url = f"{BASE_URL}{FORM_ID}/data/?format=json"
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200:
        st.error(f"Error {r.status_code} al obtener JSON desde Kobo.")
        return []
    payload = r.json()
    results = payload.get("results", [])
    return results


def find_attachment_url(record: Dict[str, Any], filename_or_value: str) -> Optional[str]:
    """Busca en _attachments el URL de descarga para el nombre de archivo dado."""
    if not filename_or_value:
        return None
    attachments = record.get("_attachments", [])
    # Algunos campos guardan el 'filename' directamente; otras veces incluyen ruta.
    for att in attachments:
        fname = att.get("filename", "")
        dl = att.get("download_url", "")
        if not dl:
            continue
        full_url = f"https://eu.kobotoolbox.org{dl}"  # completar dominio
        # Coincidencia por filename exacto o incluida en la URL
        if filename_or_value == fname or filename_or_value in full_url:
            return full_url
    return None


def download_audio_bytes(url: str) -> Optional[bytes]:
    """Descarga el archivo de audio protegido usando el token."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            return r.content
        else:
            return None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str = "base"):
    """Carga el modelo Whisper (cacheado). model_size: tiny|base|small|medium|large."""
    return whisper.load_model(model_size)


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """
    Carga un pipeline multilingüe para análisis de sentimiento.
    Usamos 'cardiffnlp/twitter-xlm-roberta-base-sentiment' (positivo/negativo/neutral).
    """
