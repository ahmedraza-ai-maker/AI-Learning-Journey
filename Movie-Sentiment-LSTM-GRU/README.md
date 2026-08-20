# Movie Sentiment Analysis — LSTM vs GRU

A Deep Learning project that performs movie sentiment classification using LSTM and GRU neural networks.

## Project Overview

This project uses the IMDB movie review dataset to classify movie reviews as Positive or Negative.

Two Recurrent Neural Network architectures are implemented:

- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

The project compares the performance of both models on the same dataset.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- LSTM
- GRU
- Natural Language Processing (NLP)

## Dataset

The project uses the IMDB movie review dataset available through TensorFlow/Keras.

The dataset contains:

- 25,000 training reviews
- 25,000 testing reviews
- Binary sentiment labels:
  - 0 = Negative
  - 1 = Positive

## Project Workflow

1. Load the IMDB dataset
2. Limit vocabulary to 10,000 words
3. Pad reviews to a fixed length of 200
4. Build an LSTM model
5. Train and evaluate the LSTM model
6. Build a GRU model
7. Train and evaluate the GRU model
8. Compare LSTM and GRU accuracy
9. Test the model on a custom movie review
10. Save both trained models

## Models

### LSTM

The LSTM model contains:

- Embedding layer
- LSTM layer with 64 units
- Dense output layer with sigmoid activation

### GRU

The GRU model contains:

- Embedding layer
- GRU layer with 64 units
- Dense output layer with sigmoid activation

## Example

Input:

> This movie was amazing and I really enjoyed it.

Output:

```text
Positive