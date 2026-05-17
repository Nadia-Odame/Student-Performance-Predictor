# 🎓 Student Performance Predictor

> A beginner-to-intermediate Machine Learning web application that predicts whether a student will perform at **Low**, **Medium**, or **High** level — built with Python, Scikit-learn, Streamlit, and Plotly.

---

## ✨ Features

| | Detail |
|---|---|
| 🤖 **ML Model** | Random Forest (200 estimators, balanced classes) |
| 📊 **Accuracy** | 92% on held-out test set |
| 🌐 **Web UI** | Streamlit dashboard with dark theme |
| 📈 **Visualisations** | Feature importance, confusion matrix, scatter plots, histograms |
| ⚡ **Live prediction** | Updates instantly as you move the sliders |
| 📁 **Dataset** | 250 synthetic but realistic student records |

---

## 📁 Project Structure

All files live **flat in one folder** — no subfolders required.

```
Student Performance Predictor/
│
├── app.py                    ← Streamlit web application (run this)
├── train_model.py            ← ML training pipeline (run this first)
├── generate_dataset.py       ← Regenerates students.csv if needed
│
├── students.csv              ← 250-row dataset
│
├── performance_model.pkl     ← Created by train_model.py
├── label_encoder.pkl         ← Created by train_model.py
├── model_metrics.pkl         ← Created by train_model.py
│
├── requirements.txt
└── README.md
```

> The three `.pkl` files are created automatically when you run `train_model.py`. You will not see them until you do.

---

## ⚙️ Setup — Step by Step

### Prerequisites
- Python 3.9 or later — check with `python3 --version`
- pip — check with `pip3 --version`

---

### Step 1 — Install dependencies

Open a terminal inside the project folder and run:

```bash
pip3 install -r requirements.txt
```

---

### Step 2 — Train the model

```bash
python3 train_model.py
```

This does everything in one go — loads the dataset, trains the model, evaluates it, and saves the three `.pkl` files.

**Expected output:**
```
======================================================
   Student Performance Predictor -- Model Training
======================================================
  Found students.csv
  250 rows loaded | classes: {'Medium': 90, 'Low': 80, 'High': 80}
  Train: 200  |  Test: 50
  Training Random Forest ...
  Accuracy: 92.0%
  ...
  Saved: performance_model.pkl
  Saved: label_encoder.pkl
  Saved: model_metrics.pkl

Done! Now run:  streamlit run app.py
```

---

### Step 3 — Launch the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

### Day-to-day usage

You only need to run `train_model.py` **once**. After that, just:

```bash
streamlit run app.py
```

Stop the app anytime with **Ctrl + C** in the terminal.

---

## 📊 Dataset

| Feature | Range | Description |
|---|---|---|
| `study_hours` | 0 – 12 | Hours spent studying per day |
| `attendance` | 30 – 100 | Class attendance percentage |
| `sleep_hours` | 3 – 12 | Hours of sleep per night |
| `previous_score` | 0 – 100 | Score from the previous exam |
| `performance` | Low / Medium / High | **Target variable** |

250 rows split across three realistic performance clusters:

| Class | Rows | Typical profile |
|---|---|---|
| Low | 80 | ~2h study, ~58% attendance, score ~42 |
| Medium | 90 | ~4.5h study, ~73% attendance, score ~62 |
| High | 80 | ~7.5h study, ~89% attendance, score ~84 |

---

## 🧠 ML Pipeline

```
students.csv
     │
     ▼
pandas DataFrame
     │
     ▼
LabelEncoder  ──  Low → 1  /  Medium → 2  /  High → 0
     │
     ▼
train_test_split  ──  80% train  /  20% test  (stratified)
     │
     ▼
RandomForestClassifier
  • n_estimators  = 200
  • max_depth     = 10
  • class_weight  = "balanced"
  • random_state  = 42
     │
     ▼
Evaluate  ──  accuracy_score + classification_report
     │
     ▼
joblib.dump  ──  saves .pkl files
     │
     ▼
Streamlit app loads .pkl  ──  live slider predictions
```

---

## 📈 Model Performance

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| High | 0.94 | 0.94 | 0.94 | 16 |
| Low | 0.89 | 1.00 | 0.94 | 16 |
| Medium | 0.94 | 0.83 | 0.88 | 18 |
| **Overall** | | | **0.92** | **50** |

### Feature Importances

| Feature | Importance |
|---|---|
| `study_hours` | ~38% |
| `previous_score` | ~38% |
| `attendance` | ~19% |
| `sleep_hours` | ~5% |

---

## 🌐 Deployment

### Option A — Streamlit Community Cloud (Free, Recommended)

1. Push the entire project folder to a **public GitHub repository**
   - Include the `.pkl` files: `git add *.pkl`
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**
4. Set:
   - Repository: `yourusername/student-performance-predictor`
   - Branch: `main`
   - Main file: `app.py`
5. Click **"Deploy"** — live in ~2 minutes

---

### Option B — Render

1. Create a free account at [render.com](https://render.com)
2. New → **Web Service** → connect your GitHub repo
3. Set:
   ```
   Build Command : pip install -r requirements.txt && python3 train_model.py
   Start Command : streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Click **"Create Web Service"**

---

### Option C — Docker

Create a `Dockerfile` in the project folder:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt && python3 train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

Build and run:

```bash
docker build -t student-predictor .
docker run -p 8501:8501 student-predictor
```

---

## 🔧 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: performance_model.pkl` | Training script not run yet | Run `python3 train_model.py` |
| `command not found: python` | macOS uses `python3` | Use `python3` instead of `python` |
| `ModuleNotFoundError: streamlit` | Dependencies not installed | Run `pip3 install -r requirements.txt` |
| `No such file: train_model.py` | Wrong folder in terminal | `cd` into the project folder first |
| Plotly color error (8-digit hex) | Older Plotly version | Run `pip3 install --upgrade plotly` |

---

## 🚀 Ideas to Extend This Project

- **More features** — internet access, extracurricular activities, study group participation
- **Compare models** — try Logistic Regression or Gradient Boosting alongside Random Forest
- **SHAP explanations** — show exactly why the model made each prediction
- **Batch prediction** — upload a CSV and predict performance for a whole class at once
- **Export PDF report** — generate a downloadable summary for each student

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| pandas | ≥ 1.5 | Data loading and manipulation |
| numpy | ≥ 1.23 | Numerical operations |
| scikit-learn | ≥ 1.2 | ML model and metrics |
| joblib | ≥ 1.2 | Saving and loading model files |
| streamlit | ≥ 1.28 | Web application framework |
| plotly | ≥ 5.15 | Interactive charts |

---

## 📄 Licence

MIT — free to use, modify, and share.
