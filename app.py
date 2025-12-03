
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Conectividad SIG – Dashboard", layout="wide")
st.title("📊 Conectividad y Habilidades Digitales – Dashboard")

uploaded_file = st.file_uploader("Sube el CSV procesado (resultados_encuesta.csv)", type=["csv"]) 
if not uploaded_file:
    st.info("Carga el archivo exportado desde Colab para ver el dashboard.")
else:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("Filtros")
    preguntas = st.sidebar.multiselect("Pregunta", sorted(df.get("pregunta", pd.Series()).dropna().unique().tolist()))
    municipios = st.sidebar.multiselect("Municipio", sorted(df.get("municipio", pd.Series()).dropna().unique().tolist()))

    if preguntas:
        df = df[df["pregunta"].isin(preguntas)]
    if municipios:
        df = df[df["municipio"].isin(municipios)]

    st.write("### Vista previa", df.head())

    if "sentimiento" in df.columns:
        st.subheader("Distribución de sentimientos")
        fig, ax = plt.subplots(figsize=(6,4))
        order = ["Negativo","Neutro","Positivo","Sin texto"]
        sns.countplot(x=df["sentimiento"], order=order, palette="coolwarm", ax=ax)
        st.pyplot(fig)

    if "transcripcion" in df.columns:
        st.subheader("Nube de palabras")
        texto = " ".join(df["transcripcion"].dropna().astype(str))
        if texto.strip():
            wc = WordCloud(width=1000, height=500, background_color="white").generate(texto)
            fig_wc, ax_wc = plt.subplots(figsize=(12,6))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No hay texto para generar la nube.")

    if "palabras_clave" in df.columns:
        st.subheader("Palabras clave por respuesta")
        st.dataframe(df[["pregunta","municipio","audio_filename","palabras_clave","transcripcion"]])
