# 🚀 Career Predictor AI

> **Discover your ideal tech career using Machine Learning.**
> Rate 12 skills, take a quiz, explore career paths, and compare roles — all powered by a trained Random Forest model with 88% accuracy.

<br>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-88%25-22c55e?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🖥️ App Modes](#️-app-modes)
- [🧠 ML Model Details](#-ml-model-details)
- [📁 Project Structure](#-project-structure)
- [⚡ Local Setup](#-local-setup)
- [🚀 Deploy on Streamlit Cloud](#-deploy-on-streamlit-cloud)
- [🔁 Retrain the Model](#-retrain-the-model)
- [🛠️ Tech Stack](#️-tech-stack)
- [🐛 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Live Prediction** | Sliders update the ML prediction instantly — no button needed |
| 🧩 **Skill Quiz** | 6 personality questions auto-build your skill profile |
| 🗺️ **Career Explorer** | Deep-dive into any of 10 tech careers with skill gap analysis |
| ⚖️ **Career Comparison** | Side-by-side radar chart & skill breakdown of any two careers |
| 📊 **Visualisations** | Radar charts, probability bars, animated confidence meters |
| 💰 **Salary & Demand** | Real salary ranges (India) + job market demand per career |
| 🔄 **Auto-Build Model** | App trains itself from Excel if `.joblib` files are missing |
| 🎨 **Custom Dark UI** | Animated orbs, gradient text, custom HTML/CSS throughout |

---

## 🖥️ App Modes

### 🚀 Tab 1 — Live Predictor
The main prediction screen. Drag any of the 12 skill sliders and the Random Forest model instantly recomputes your best-fit career, confidence %, a radar chart (You vs Ideal), and a full probability bar chart for all 10 careers.

- **4 Quick Presets** — Analyst / Creative / Leader / Engineer
- **Insight cards** — Top Skill, Average Score, Skill to Grow
- **Reset button** — wipe all sliders back to 5

### 🎯 Tab 2 — Skill Quiz
6 multiple-choice questions that probe your personality and working style. Each answer quietly boosts the relevant skill scores. After question 6 you get a full result card + radar + top-5 match list. Use **"Copy Scores → Live Predictor"** to port your quiz profile into Tab 1.

### 🗺️ Tab 3 — Career Explorer
Click any of the 10 career cards to unlock:
- Full career description, salary range, and market demand
- **Skill gap bars** — your current score vs the ideal target, colour-coded (✅ on target / 🟡 close / 🔴 gap)
- **Dual radar overlay** — You vs Ideal

### ⚖️ Tab 4 — Compare Careers
Pick any two careers from dropdown menus to see:
- Header cards with salary & demand
- **Radar overlay** of both skill profiles
- **Row-by-row skill table** with gradient bars and 🔺🔻 direction arrows
- **Career Similarity Score** — % overlap with plain-English interpretation

---

## 🧠 ML Model Details

### Algorithm

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees |
| Max Depth | 12 |
| Preprocessing | StandardScaler (inside Pipeline) |
| Serialisation | **Joblib** (`compress=3`) |
| Model size | ~658 KB |
| Test Accuracy | **~88%** |
| CV Accuracy | **~85% ± 2.6%** (5-fold) |
| Training samples | 1 000 |
| Features | 12 skill scores (each 1–10) |
| Target classes | 10 tech career paths |

### 12 Input Features

| # | Feature | Description |
|---|---|---|
| 1 | `math` | Mathematical aptitude |
| 2 | `logic` | Logical & analytical thinking |
| 3 | `creativity` | Creative & design thinking |
| 4 | `communication` | Verbal & written communication |
| 5 | `leadership` | Leadership & influence |
| 6 | `problem_solving` | General problem-solving ability |
| 7 | `programming` | Coding & software development |
| 8 | `data_analysis` | Data handling & interpretation |
| 9 | `design` | UI/UX & visual design |
| 10 | `networking` | Computer networking knowledge |
| 11 | `management` | Project & people management |
| 12 | `writing` | Technical & creative writing |

### 10 Career Paths Predicted

| Icon | Career | Avg Salary (India) | Demand |
|---|---|---|---|
| 💻 | Software Engineer | ₹8L – ₹35L | 🔥 Very High |
| 📊 | Data Scientist | ₹10L – ₹40L | 🔥 Very High |
| 🌐 | Web Developer | ₹5L – ₹22L | ✅ High |
| 🛡️ | Cybersecurity Analyst | ₹8L – ₹30L | 🔥 Very High |
| 🤖 | AI/ML Engineer | ₹12L – ₹50L | 🚀 Explosive |
| 🗄️ | Database Administrator | ₹6L – ₹25L | ✅ Moderate |
| 🔗 | Network Engineer | ₹5L – ₹20L | ✅ Moderate |
| 🎨 | UI/UX Designer | ₹5L – ₹25L | ✅ High |
| 📋 | Product Manager | ₹12L – ₹45L | ✅ High |
| ⚙️ | DevOps Engineer | ₹8L – ₹32L | 🔥 Very High |

### Model Pipeline

```
Input (12 floats)
      │
      ▼
 StandardScaler        ← normalise each skill to zero mean, unit variance
      │
      ▼
 RandomForestClassifier (100 trees, depth ≤ 12)
      │
      ▼
 predict_proba()       ← returns probability for all 10 career classes
      │
      ▼
 LabelEncoder.inverse_transform()   ← map class index → career name
```

---

## 📁 Project Structure

```
career_predictor/
│
├── app.py                    # ★ Main Streamlit app (4 interactive tabs)
├── train_model.py            # Standalone model training script
├── create_dataset.py         # Generates the Excel training dataset
│
├── career_dataset.xlsx       # Excel dataset — 1000 rows × 13 columns
│
├── models/
│   ├── career_model.joblib   # Trained RF pipeline (compressed, ~658 KB)
│   └── label_encoder.joblib  # Sklearn LabelEncoder for career names
│
├── .streamlit/
│   └── config.toml           # Dark theme + server config
│
├── requirements.txt          # Python package dependencies
├── packages.txt              # System packages (empty — none needed)
├── .gitattributes            # Marks .joblib/.xlsx as binary for Git
├── .gitignore                # Ignores __pycache__, .env, venv, etc.
└── README.md                 # This file
```

---

## ⚡ Local Setup

### Prerequisites

- Python 3.9 or higher
- `pip` package manager
- Git

### Step-by-step

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/career-predictor.git
cd career-predictor

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install all dependencies
pip install -r requirements.txt

# 4. (Optional) Regenerate the Excel dataset from scratch
python create_dataset.py

# 5. (Optional) Retrain and re-save the model
python train_model.py

# 6. Launch the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The app will auto-train the model from the Excel file on first launch if the `.joblib` files are missing.

---

## 🚀 Deploy on Streamlit Cloud

### Step 1 — Push your project to GitHub

```bash
# Initialise git (skip if already done)
git init
git add .
git commit -m "🚀 Initial commit — Career Predictor AI"
git branch -M main

# Create the remote repo on GitHub first, then:
git remote add origin https://github.com/YOUR_USERNAME/career-predictor.git
git push -u origin main
```

> ⚠️ **Important:** Make sure `models/career_model.joblib`, `models/label_encoder.joblib`, and `career_dataset.xlsx` are all committed. The `.gitattributes` file ensures Git treats them as binary and does not corrupt them.

### Step 2 — Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with your **GitHub** account
3. Click **"New app"**
4. Fill in the form:
   - **Repository:** `YOUR_USERNAME/career-predictor`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

Streamlit Cloud will automatically install everything in `requirements.txt`. The first boot takes ~60 seconds.

> ✅ **No secrets or environment variables needed.** The app is fully self-contained.

### Step 3 — Verify deployment

Once live, open your app URL and confirm:
- The hero section loads with animated gradient text
- Tab 1 sliders update the prediction in real time
- Tab 2 quiz progresses through all 6 questions
- Tab 3 career cards open a skill gap panel
- Tab 4 comparison renders dual radar charts

---

## 🔁 Retrain the Model

If you want to update the dataset, add more samples, or tune hyperparameters:

```bash
# 1. Regenerate the Excel dataset (optional)
python create_dataset.py          # produces career_dataset.xlsx

# 2. Retrain the model with new data
python train_model.py             # produces models/*.joblib

# 3. Commit and push the updated artefacts
git add career_dataset.xlsx models/
git commit -m "♻️ Retrain model with updated dataset"
git push
```

On Streamlit Cloud, click **"Reboot app"** from the management panel after pushing.

### Tuning hyperparameters

Open `train_model.py` and edit the `RandomForestClassifier` block:

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=100,   # increase for higher accuracy (slower)
        max_depth=12,       # increase for more complex decision boundaries
        random_state=42,
        n_jobs=-1
    ))
])
```

After changing, re-run `python train_model.py` to rebuild the `.joblib` files.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI Framework** | [Streamlit](https://streamlit.io) | Web app, tabs, sliders, state |
| **Styling** | Custom HTML + CSS | Dark theme, animations, gradients |
| **ML Model** | [scikit-learn](https://scikit-learn.org) | Random Forest, Pipeline, LabelEncoder |
| **Serialisation** | [Joblib](https://joblib.readthedocs.io) | Save/load trained model pipeline |
| **Dataset** | Excel `.xlsx` via [openpyxl](https://openpyxl.readthedocs.io) | Training data storage |
| **Data handling** | [Pandas](https://pandas.pydata.org) + [NumPy](https://numpy.org) | Feature engineering, predictions |
| **Charts** | [Plotly](https://plotly.com/python/) | Radar charts, bar charts |
| **Deployment** | [Streamlit Community Cloud](https://share.streamlit.io) | Free hosting via GitHub |
| **Version control** | [GitHub](https://github.com) | Source code + model storage |

---

## 🐛 Troubleshooting

### `FileNotFoundError` for `.joblib` on Streamlit Cloud

**Cause:** Relative paths like `"models/career_model.joblib"` don't resolve correctly when Streamlit Cloud changes the working directory.

**Fix (already applied):** `app.py` uses `os.path.abspath(__file__)` to anchor all paths to the script's own directory:

```python
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "career_model.joblib")
```

**Auto-heal:** If the `.joblib` files are missing entirely, `app.py` automatically trains from `career_dataset.xlsx` on first run (~5 seconds).

---

### Model files not pushed to GitHub

**Symptom:** `models/` folder shows in `.gitignore` or files are 0 bytes after push.

**Fix:** Confirm `.gitattributes` is committed and run:

```bash
git rm --cached models/career_model.joblib models/label_encoder.joblib
git add models/career_model.joblib models/label_encoder.joblib
git commit -m "fix: force-add binary model files"
git push
```

---

### `ModuleNotFoundError` on Streamlit Cloud

**Fix:** Ensure all packages are listed in `requirements.txt`:

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
openpyxl>=3.1.0
plotly>=5.18.0
xlsxwriter>=3.1.0
```

---

### Slow first load on Streamlit Cloud

**Cause:** Cold start + `@st.cache_resource` warming up.

**Expected behaviour:** First load takes 20–60 seconds. All subsequent loads are instant because the model is cached in memory.

---

### Sliders not reflecting preset values

**Fix:** Presets update `st.session_state` keys and call `st.rerun()`. If a slider key collision occurs, clear browser cache or rename slider keys in `app.py`.

---

## 📄 License

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<div align="center">

**Built with ❤️ using Streamlit · scikit-learn · Joblib · Excel**

[⭐ Star this repo](https://github.com/YOUR_USERNAME/career-predictor) · [🐛 Report a Bug](https://github.com/YOUR_USERNAME/career-predictor/issues) · [💡 Request a Feature](https://github.com/YOUR_USERNAME/career-predictor/issues)

</div>
