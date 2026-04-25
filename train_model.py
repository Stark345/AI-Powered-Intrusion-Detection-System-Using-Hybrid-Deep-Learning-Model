import os
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import pickle

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_DIR = "wlasl_landmarks"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

import os
import numpy as np

DATA_DIR = "wlasl_landmarks"

# Debug: check filenames before loading
print("First 20 filenames and extracted labels:")
for f in os.listdir(DATA_DIR)[:20]:
    print(f, "->", f.split('_')[0])

landmarks = []
labels = []

# Main loading loop
for f in os.listdir(DATA_DIR):
    if f.endswith(".npy"):
        file_path = os.path.join(DATA_DIR, f)
        data = np.load(file_path)

        if len(data.shape) == 2 and data.shape[0] > 10:
            landmarks.append(data.astype(np.float32))
            label = f.split('_')[0]
            labels.append(label)

# Load landmark sequences
landmarks = []
labels = []

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Folder '{DATA_DIR}' not found.")

for f in os.listdir(DATA_DIR):
    if f.endswith(".npy"):
        file_path = os.path.join(DATA_DIR, f)
        try:
            data = np.load(file_path)

            # Expect shape: (timesteps, features)
            if len(data.shape) == 2 and data.shape[0] > 10:
                landmarks.append(data.astype(np.float32))
                label = f.split("_")[0]
                labels.append(label)
            else:
                print(f"Skipping {f}: invalid shape {data.shape}")
        except Exception as e:
            print(f"Error loading {f}: {e}")

print(f"Loaded {len(landmarks)} valid sequences.")

if len(landmarks) == 0:
    raise ValueError("No valid landmark sequences found.")

# Count class samples
label_counts = Counter(labels)
print("\nClass distribution before filtering:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Filter rare classes
MIN_SAMPLES_PER_CLASS = 2

filtered_landmarks = []
filtered_labels = []

for seq, label in zip(landmarks, labels):
    if label_counts[label] >= MIN_SAMPLES_PER_CLASS:
        filtered_landmarks.append(seq)
        filtered_labels.append(label)

landmarks = filtered_landmarks
labels = filtered_labels

if len(landmarks) == 0:
    raise ValueError("No sequences left after filtering rare classes.")

print(f"\nRemaining sequences after filtering: {len(landmarks)}")
print(f"Remaining classes: {len(set(labels))}")

# Recount after filtering
label_counts = Counter(labels)
print("\nClass distribution after filtering:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Pad sequences
max_len = max(seq.shape[0] for seq in landmarks)
feature_dim = landmarks[0].shape[1]

X = np.array([
    np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), mode='constant')
    for seq in landmarks
], dtype=np.float32)

print(f"\nInput shape: {X.shape}")

# Encode labels
le = LabelEncoder()
y_labels = le.fit_transform(labels)
num_classes = len(le.classes_)
y = tf.keras.utils.to_categorical(y_labels, num_classes=num_classes)

print(f"Number of classes: {num_classes}")

# Check if stratified split is possible
min_class_count = min(Counter(y_labels).values())
print(f"Minimum samples in any class: {min_class_count}")

if min_class_count >= 2:
    print("\nUsing stratified train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y_labels
    )
else:
    print("\nUsing normal train-test split (stratify disabled)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Build model
model = Sequential([
    Masking(mask_value=0.0, input_shape=(max_len, feature_dim)),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_sign_model.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# Save final model
final_model_path = os.path.join(MODEL_DIR, "final_sign_model.keras")
model.save(final_model_path)

# Save label encoder
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
with open(label_encoder_path, "wb") as f:
    pickle.dump(le, f)

print("\nModel saved successfully.")
print(f"Best model: {os.path.join(MODEL_DIR, 'best_sign_model.keras')}")
print(f"Final model: {final_model_path}")
print(f"Label encoder: {label_encoder_path}")
print("Ready for webcam demo.")