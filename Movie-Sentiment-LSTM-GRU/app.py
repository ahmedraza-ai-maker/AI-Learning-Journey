import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("Movie Sentiment Analysis")
st.write("LSTM vs GRU Sentiment Classifier")

# -----------------------------
# Settings
# -----------------------------
VOCAB_SIZE = 10000
MAX_LENGTH = 200


# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    (x_train, y_train), _ = imdb.load_data(num_words=VOCAB_SIZE)

    # Smaller training set for Streamlit deployment
    x_train = x_train[:5000]
    y_train = y_train[:5000]

    x_train = pad_sequences(
        x_train,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return x_train, y_train


# -----------------------------
# Build Model
# -----------------------------
@st.cache_resource
def train_model(model_type):

    x_train, y_train = load_data()

    model = Sequential([
        Embedding(VOCAB_SIZE, 64, input_length=MAX_LENGTH)
    ])

    if model_type == "LSTM":
        model.add(LSTM(64))
    else:
        model.add(GRU(64))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )

    return model


# -----------------------------
# Model Selection
# -----------------------------
model_type = st.selectbox(
    "Choose Model",
    ["LSTM", "GRU"]
)

st.info(
    "The first prediction may take some time because the selected "
    "model is trained when the app starts."
)

model = train_model(model_type)


# -----------------------------
# Review Input
# -----------------------------
review = st.text_area(
    "Enter a movie review:",
    placeholder="Example: This movie was amazing and I really enjoyed it."
)


# -----------------------------
# Convert Review to Sequence
# -----------------------------
def preprocess_review(text):

    word_index = imdb.get_word_index()

    words = text.lower().split()

    encoded = []

    for word in words:
        encoded.append(word_index.get(word, 2) + 3)

    encoded = pad_sequences(
        [encoded],
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    return encoded


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Sentiment"):

    if not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        processed_review = preprocess_review(review)

        prediction = model.predict(
            processed_review,
            verbose=0
        )[0][0]

        confidence = prediction * 100

        if prediction >= 0.5:

            st.success("Positive Sentiment")

            st.write(
                f"Positive probability: {confidence:.2f}%"
            )

        else:

            st.error("Negative Sentiment")

            st.write(
                f"Negative probability: {100 - confidence:.2f}%"
            )


# -----------------------------
# Project Information
# -----------------------------
st.divider()

st.subheader("About this project")

st.write(
    "This project demonstrates movie sentiment classification "
    "using Recurrent Neural Networks with LSTM and GRU architectures."
)

st.write(
    "Dataset: IMDB Movie Reviews"
)