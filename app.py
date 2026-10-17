"""
Spotify Express - Recomendador de Estilo Musical
------------------------------------------------
Aplicação corporativa em Streamlit: o colaborador informa seu nível de
"Energia" e "Tristeza" e uma rede neural (TensorFlow) indica o estilo musical
mais adequado ao humor atual: Rock, Funk, Lofi, Pop ou Sertanejo.

Rodar localmente:
    pip install -r requirements.txt
    streamlit run app.py

Render (Web Service):
    Build Command: pip install -r requirements.txt
    Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
SEED = 42

# Cada estilo é definido por um "centro" no plano (energia, tristeza), de 0 a 100,
# mais uma descrição e um título de playlist sugerido para o colaborador.
ESTILOS = {
    "Rock": {
        "centro": (85, 65),
        "emoji": "🎸",
        "playlist": "Rock: Descarga de Energia",
        "descricao": "Alta energia com emoção intensa. Ideal para transformar tensão em foco.",
    },
    "Funk": {
        "centro": (90, 10),
        "emoji": "🔊",
        "playlist": "Funk: Ritmo e Motivação",
        "descricao": "Muita energia e astral elevado. Ótimo para começar o dia com disposição.",
    },
    "Pop": {
        "centro": (65, 30),
        "emoji": "🎤",
        "playlist": "Pop: Equilíbrio no Expediente",
        "descricao": "Energia moderada e clima leve. Combina com tarefas variadas do dia a dia.",
    },
    "Lofi": {
        "centro": (20, 35),
        "emoji": "🎧",
        "playlist": "Lofi: Concentração Profunda",
        "descricao": "Baixa energia e ambiente calmo. Perfeito para análises e trabalho focado.",
    },
    "Sertanejo": {
        "centro": (40, 85),
        "emoji": "🪕",
        "playlist": "Sertanejo: Momento de Reflexão",
        "descricao": "Energia contida e lado mais sentimental. Boa pedida para desacelerar.",
    },
}
NOMES = list(ESTILOS.keys())  # a ordem define o índice de cada classe no modelo

st.set_page_config(page_title="Spotify Express", page_icon="🎵", layout="wide")


# ---------------------------------------------------------------------------
# Visual inspirado no Spotify (tema escuro + verde)
# ---------------------------------------------------------------------------
def aplicar_tema() -> None:
    """Injeta CSS para deixar o app com visual escuro e destaque em verde."""
    st.markdown(
        """
        <style>
            .stApp { background-color: #121212; color: #FFFFFF; }
            [data-testid="stSidebar"] { background-color: #000000; }
            h1, h2, h3, label, p, span { color: #FFFFFF; }
            .cartao {
                background: linear-gradient(135deg, #1DB954 0%, #145A32 100%);
                border-radius: 16px; padding: 32px; margin-bottom: 16px;
            }
            .cartao small { color: #E8F5E9; letter-spacing: 2px; }
            .cartao h1 { font-size: 3rem; margin: 8px 0; }
            .cartao p { font-size: 1.05rem; margin: 4px 0; }
            .rodape { color: #B3B3B3; font-size: 0.85rem; margin-top: 24px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Dados e modelo
# ---------------------------------------------------------------------------
def gerar_dados(amostras_por_estilo: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Gera exemplos sintéticos ao redor do centro de cada estilo."""
    rng = np.random.default_rng(SEED)
    entradas, rotulos = [], []

    for indice, nome in enumerate(NOMES):
        centro = np.array(ESTILOS[nome]["centro"], dtype="float32")
        # Espalha pontos em torno do centro (desvio de 10 pontos) para simular
        # a variação natural de humor entre pessoas.
        pontos = rng.normal(centro, 10, size=(amostras_por_estilo, 2))
        entradas.append(np.clip(pontos, 0, 100))
        rotulos.append(np.full(amostras_por_estilo, indice))

    x = np.vstack(entradas).astype("float32") / 100.0  # escala 0-1 ajuda o treino
    y = np.concatenate(rotulos).astype("int32")
    return x, y


@st.cache_resource  # treina uma única vez, mesmo com várias interações no slider
def treinar_modelo() -> tf.keras.Model:
    """Treina uma rede neural pequena que classifica o humor em um estilo."""
    tf.keras.utils.set_random_seed(SEED)
    x, y = gerar_dados()

    modelo = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),                        # energia, tristeza
            tf.keras.layers.Dense(16, activation="relu"),      # camada oculta 1
            tf.keras.layers.Dense(16, activation="relu"),      # camada oculta 2
            tf.keras.layers.Dense(len(NOMES), activation="softmax"),  # 1 prob. por estilo
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # rótulos como números inteiros
        metrics=["accuracy"],
    )
    modelo.fit(x, y, epochs=60, batch_size=32, validation_split=0.2, verbose=0)
    return modelo


def indicar_musica_pelo_humor(modelo: tf.keras.Model, energia: int, tristeza: int) -> pd.Series:
    """
    Objetivo: indicar o estilo musical com base no humor atual do usuário.
    Retorna a probabilidade (em %) de cada estilo, já ordenada do maior para o menor.
    """
    entrada = np.array([[energia, tristeza]], dtype="float32") / 100.0
    probabilidades = modelo.predict(entrada, verbose=0)[0] * 100
    return pd.Series(probabilidades, index=NOMES).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
aplicar_tema()
modelo = treinar_modelo()

# Coluna lateral (sidebar): controles de humor, no estilo "biblioteca" do Spotify
with st.sidebar:
    st.markdown("## 🎵 Spotify Express")
    st.caption("Recomendação musical corporativa")
    st.markdown("### Como você está agora?")
    energia = st.slider("⚡ Energia", 0, 100, 50, help="0 = sem energia, 100 = muito energizado")
    tristeza = st.slider("🌧️ Tristeza", 0, 100, 50, help="0 = muito bem, 100 = muito triste")
    st.markdown(
        "<p class='rodape'>Uso interno. As respostas não são armazenadas.</p>",
        unsafe_allow_html=True,
    )

# Área principal: resultado
resultado = indicar_musica_pelo_humor(modelo, energia, tristeza)
melhor = resultado.index[0]
info = ESTILOS[melhor]

st.markdown("# Sua recomendação")

coluna_principal, coluna_detalhes = st.columns([3, 2], gap="large")

with coluna_principal:
    st.markdown(
        f"""
        <div class="cartao">
            <small>ESTILO RECOMENDADO</small>
            <h1>{info['emoji']} {melhor}</h1>
            <p><b>Playlist sugerida:</b> {info['playlist']}</p>
            <p>{info['descricao']}</p>
            <p><b>Confiança:</b> {resultado.iloc[0]:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with coluna_detalhes:
    st.markdown("### Compatibilidade por estilo")
    st.bar_chart(resultado.rename("Compatibilidade (%)"), color="#1DB954")
    st.caption("Distribuição de probabilidade calculada pelo modelo.")
