# 🛡️ AI-Powered Intrusion Detection System (Hybrid Deep Learning & XAI)

> **Enterprise Network Threat Detection & Explainable Cybersecurity Framework**  
> Combines **1D Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)**, **SMOTE Data Rebalancing**, and **Explainable AI (SHAP & LIME)** to detect and explain sophisticated cyber attacks on benchmark network telemetry (NSL-KDD / CICIDS).

---

## 🌟 Architectural Highlights

- **Hybrid Deep Learning Core**:
  - **Conv1D Layers**: Extracts local, high-frequency spatial signatures across 41 packet flow attributes.
  - **LSTM Networks**: Captures long-range temporal attack sequences and multi-step advanced persistent threats (APTs).
- **SMOTE Class Rebalancing**: Solves the critical real-world problem of imbalanced network telemetry where malicious anomalies represent <1% of legitimate traffic.
- **Explainable AI (XAI) for SOC Analysts**:
  - **SHAP (SHapley Additive exPlanations)**: Global feature importance quantification showing systemic vulnerability vectors.
  - **LIME (Local Interpretable Model-agnostic Explanations)**: Per-packet attribution providing incident responders with root-cause transparency.
- **Superior Benchmark Performance**: Evaluated against the rigorous **NSL-KDD Test Set (`KDDTest+`)**, achieving a **0.9268 ROC-AUC** and **91% Attack Precision**.

---

## 🏛️ Deep Learning Model Topology

```
Input: 41-Dimensional Network Packet Attributes
                      │
                      ▼
[ ColumnTransformer: One-Hot Encoding + StandardScaler ]
                      │
                      ▼
[ Synthetic Minority Over-sampling Technique (SMOTE) ]
                      │
                      ▼
[ 1D-CNN Block 1: Conv1D(32, k=3) + BatchNorm + MaxPool1D(2) ]  ──> Captures spatial burst patterns
                      │
                      ▼
[ 1D-CNN Block 2: Conv1D(64, k=3) + BatchNorm + MaxPool1D(2) ]  ──> Deep hierarchical abstraction
                      │
                      ▼
[ Recurrent Block: LSTM(64) + Dropout(0.3) ]                   ──> Temporal sequence dependencies
                      │
                      ▼
[ Dense Decision Block: Dense(64) + Dense(32) + Dropout(0.2) ]
                      │
                      ▼
[ Sigmoid Classification Output: Normal (0) vs Attack (1) ]
```

---

## 📊 Benchmark Evaluation Metrics

Evaluated on the full unseen **NSL-KDD Test Set (`KDDTest+`)**:

| Metric | Score | Analytical Interpretation |
|---|---|---|
| **ROC-AUC** | **0.9268 (92.7%)** | High discriminative capability across varying threshold trade-offs |
| **Attack Precision** | **0.91 (91.0%)** | Extremely low false positive rate; minimizes SOC alert fatigue |
| **Attack F1-Score** | **0.75** | Balanced harmonic precision-recall representation on difficult test splits |
| **Normal Recall** | **0.92 (92.0%)** | Permissive baseline transmission with zero operational disruption |

---

## 🔍 Explainable AI (XAI) in Action

Traditional deep learning models act as opaque "black boxes," making security engineers hesitant to trust automated blocking decisions. This framework integrates **SHAP** and **LIME** to output transparent feature weights for every prediction:

```
📡 Packet Inspection: [SYN FLOOD ATTACK DETECTED]
   Confidence: 98.2% | Severity: HIGH
   Top Attack Indicators (XAI Attributions):
     1. serror_rate (+0.46)   ──> 100% SYN connection attempts unacknowledged
     2. count (+0.32)         ──> 256 connection requests in 2-second sliding window
     3. flag=S0 (+0.28)       ──> Connection initiation without complete three-way handshake
```

---

## 📁 Repository Structure

```
AI-Powered-Intrusion-Detection-System-Using-Hybrid-Deep-Learning-Model/
├── src/
│   ├── __init__.py
│   ├── model.py              # Hybrid CNN-LSTM Neural Network architecture
│   ├── preprocess.py         # 41-feature ColumnTransformer & SMOTE pipeline
│   └── explain.py            # SHAP & LIME interpretability engine
├── predict.py                # Standalone CLI threat assessment tool
├── train.py                  # End-to-end model training script
├── sample_traffic.json       # Benchmark normal and attack packet telemetry
├── IDS.ipynb                 # Interactive Jupyter notebook with visualization charts
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignores raw datasets (*.arff, *.txt), venv, models
└── README.md                 # Project documentation
```

---

## 🚀 Quickstart & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Stark345/AI-Powered-Intrusion-Detection-System-Using-Hybrid-Deep-Learning-Model.git
cd AI-Powered-Intrusion-Detection-System-Using-Hybrid-Deep-Learning-Model
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run CLI Threat Detection Demo
```bash
python predict.py --demo
```

### 5. Train the Model from Scratch
```bash
python train.py
```

### 6. Interactive Jupyter Notebook
Open `IDS.ipynb` in Jupyter Notebook or Google Colab to view real-time training graphs, confusion matrix heatmaps, and interactive SHAP force plots.

---

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow / Keras (Conv1D, LSTM, BatchNormalization)
- **Machine Learning & Preprocessing**: Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Explainable AI (XAI)**: SHAP (Shapley Additive exPlanations), LIME
- **Data Engineering**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

---

## 👨‍💻 Developed By

**S Jaichandran** — *AI & Full-Stack Developer*  
- Portfolio: [sjaichandran-portfolio.netlify.app](https://sjaichandran-portfolio.netlify.app/)  
- GitHub: [@Stark345](https://github.com/Stark345)  
- LinkedIn: [jaichandran-s-139a8b354](https://www.linkedin.com/in/jaichandran-s-139a8b354)
