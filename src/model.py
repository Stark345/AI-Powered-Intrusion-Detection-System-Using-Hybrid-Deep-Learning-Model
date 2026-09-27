"""
Hybrid Deep Learning Architecture for Network Intrusion Detection
Combines 1D-CNN (spatial feature extraction) + LSTM (temporal sequence dependencies)
Follows IEEE publication benchmark architecture for NSL-KDD / CICIDS datasets.
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization


def build_hybrid_model(input_features: int, learning_rate: float = 0.001) -> Sequential:
    """
    Constructs the hybrid Conv1D + LSTM neural network.
    
    Args:
        input_features: Number of preprocessed input feature dimensions.
        learning_rate: Initial Adam optimizer learning rate.
        
    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        # Stage 1: Spatial Local Feature Extraction
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_features, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Stage 2: Deep Pattern Hierarchy
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Stage 3: Temporal Sequential Memory
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        # Stage 4: Dense Decision Layers & Regularization
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),

        # Stage 5: Classification (Normal vs Malicious Attack)
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    return model
