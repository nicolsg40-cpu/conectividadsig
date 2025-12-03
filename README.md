# Conectividad SIG – Dashboard

Este proyecto implementa un flujo **100% automático** para recolectar, procesar y visualizar datos de un formulario en KoboToolbox.

## ✅ Características
- Conexión automática a **KoboToolbox** usando API.
- Descarga de audios adjuntos y **transcripción con Whisper**.
- **Análisis de sentimiento** multilingüe con modelos de Transformers.
- Extracción de **palabras clave** por respuesta.
- Dashboard interactivo en **Streamlit** con:
  - Filtros por pregunta, municipio y sentimiento.
  - Distribución de sentimientos.
  - Nube de palabras.
  - Tabla detallada con transcripciones y palabras clave.
  - Reproductor de audios.
  - Mapa con coordenadas GPS.

## 🚀 Despliegue en Streamlit Cloud

1. **Sube estos archivos a tu repositorio GitHub**:
   - `app.py` (código principal)
   - `requirements.txt` (dependencias)
   - `README.md` (este archivo)

2. **Conecta tu repo a Streamlit Cloud**:
   - Ve a [Streamlit Cloud](https://streamlit.io/cloud)
   - Inicia sesión con tu cuenta GitHub.
   - Haz clic en **New app** y selecciona el repositorio.
   - Configura:
     - **Branch**: `main`
     - **Main file path**: `app.py`
   - Haz clic en **Deploy**.

3. **Configura secretos para el token**:
   - En la app desplegada, ve a **Settings → Secrets**.
   - Agrega:
     ```toml
     API_TOKEN = "TU_TOKEN_DE_KOBO"
     FORM_ID = "ID_DEL_FORMULARIO"
     ```
   - En `app.py`, asegúrate de usar:
     ```python
     API_TOKEN = st.secrets["API_TOKEN"]
     FORM_ID = st.secrets["FORM_ID"]
     ```

## 📦 Dependencias
Incluidas en `requirements.txt`:
```
streamlit
pandas
matplotlib
seaborn
wordcloud
requests
openai-whisper
transformers
torch
```

## ▶️ Uso
Una vez desplegada la app:
- Descargará datos automáticamente desde KoboToolbox.
- Procesará audios y texto.
- Mostrará gráficos y tablas interactivas.

## 🔒 Seguridad
- Nunca subas tu token directamente en el código.
- Usa **Streamlit Secrets** para manejar credenciales.

---

**Autor:** Proyecto para mapeo digital y análisis de conectividad.
