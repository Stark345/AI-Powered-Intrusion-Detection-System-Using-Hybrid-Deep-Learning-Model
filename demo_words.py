import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import pyttsx3
import threading

# --- 1. INITIALIZE VOICE ---
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    def target():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=target, daemon=True).start()

# --- 2. LOAD AI MODEL & LABELS ---
print("🔄 Loading AI Model...")
model = tf.keras.models.load_model('models/sign_model.h5')
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

word_dict = {
    '00335.npy': 'HELLO', '00336.npy': 'THANK YOU', '00338.npy': 'GOOD MORNING',
    '00339.npy': 'HELP', '00341.npy': 'PLEASE', '00376.npy': 'YES',
    '00377.npy': 'NO', '00381.npy': 'WATER', '00382.npy': 'FOOD',
    '00384.npy': 'HUNGRY', '00414.npy': 'DOCTOR', '00415.npy': 'PAIN',
    '00416.npy': 'FAMILY', '00421.npy': 'LOVE', '00426.npy': 'HAPPY',
    '00430.npy': 'SAD', '00431.npy': 'HOME', '00433.npy': 'SCHOOL',
    '00435.npy': 'WORK', '00583.npy': 'MONEY'
}

# --- 3. CAMERA SETUP ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Standard res for better motion math
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- 4. VARIABLES & TUNED THRESHOLDS ---
seq_length = 30
sequence = deque(maxlen=seq_length)
sentence = []
current_word = "WAITING..."
current_conf = 0.0
last_idx = -1

# Based on your logs, we are setting these specifically for your environment:
CONF_THRESH = 20.0   # Lowered because your model is peaked at 24%
MOVE_GATE = 10.0     # Your movement is at 80+, so 10 is a safe "active" floor

print(f"✅ LIVE! Thresholds: Conf > {CONF_THRESH}%, Move > {MOVE_GATE}")

while True:
    ret, frame = cap.read()
    if not ret: break

    # FEATURE EXTRACTION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Motion Math
    m_val = np.mean(np.abs(gray - np.roll(gray, 1, axis=(0,1))))
    act = np.std(gray)
    brt = np.mean(gray)
    pos = min(len(sequence)+1, seq_length)
    
    features = np.array([m_val, act, brt, pos])
    sequence.append(features)

    # PREDICTION LOGIC
    if len(sequence) == seq_length:
        seq_array = np.array(list(sequence))
        avg_move = np.mean(seq_array[:, 0]) 
        
        pred = model.predict(np.expand_dims(seq_array, 0), verbose=0)
        word_idx = np.argmax(pred)
        conf = np.max(pred) * 100

        # PRINT TO CONSOLE
        print(f"Guess: {le.classes_[word_idx]} | Conf: {conf:.1f}% | Move: {avg_move:.1f}")

        # DISPLAY LOGIC
        if conf > CONF_THRESH and avg_move > MOVE_GATE:
            if word_idx != last_idx:
                video_id = le.classes_[word_idx]
                current_word = word_dict.get(video_id, video_id)
                current_conf = conf
                last_idx = word_idx
                
                if not sentence or sentence[-1] != current_word:
                    sentence.append(current_word)
                    speak(current_word)
        
        # If movement stops, reset index so we can trigger the word again
        if avg_move < 5.0:
            last_idx = -1

    # --- 5. UI DISPLAY ---
    # Draw Background Bar
    cv2.rectangle(frame, (0, 400), (640, 480), (30, 30, 30), -1)
    
    # Main Word Display
    if current_word != "WAITING...":
        cv2.putText(frame, f"{current_word}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(frame, f"{current_conf:.1f}%", (35, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "READY: MOVE NOW", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)

    # Sentence Bar
    cv2.putText(frame, f"SENTENCE: {' '.join(sentence[-4:])}", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('ASL Final Demo', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'):
        sentence = []; current_word = "WAITING..."; last_idx = -1; sequence.clear()
        print("🔄 Reset!")

cap.release()
cv2.destroyAllWindows()