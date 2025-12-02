
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Configuración de la página
st.set_page_config(page_title="Dashboard Encuesta", layout="wide")

st.title("📊 Dashboard de Encuesta sobre Conectividad y Habilidades Digitales")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube el archivo CSV procesado", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Filtros
    st.sidebar.header("Filtros")
    preguntas = st.sidebar.multiselect("Filtrar por pregunta", df["pregunta"].unique())
    genero = st.sidebar.multiselect("Filtrar por género", df["genero"].dropna().unique())
    ciudad = st.sidebar.multiselect("Filtrar por ciudad", df["ciudad"].dropna().unique())

    # Aplicar filtros
    if preguntas:
        df = df[df["pregunta"].isin(preguntas)]
    if genero:
        df = df[df["genero"].isin(genero)]
    if ciudad:
        df = df[df["ciudad"].isin(ciudad)]

    st.write("### Vista previa de datos filtrados", df.head())

    # Gráfico de sentimientos
    st.subheader("Distribución de Sentimientos")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=df["sentimiento"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Nube de palabras
    st.subheader("Nube de Palabras")
    texto_completo = " ".join(df["transcripcion"].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(texto_completo)
    fig_wc, ax_wc = plt.subplots(figsize=(10,6))
    ax_wc.imshow(wordcloud, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # Tabla de palabras clave
    if "palabras_clave" in df.columns:
        st.subheader("Palabras clave por respuesta")
        st.dataframe(df[["pregunta", "palabras_clave"]])
