# 🌸 Iris Flower Classifier

A production-ready Machine Learning web application built with **Streamlit**, **Scikit-Learn**, **Plotly**, and **Pandas** — suitable for a Data Science / ML portfolio.

---

## 📋 Features

| Page | Description |
|------|-------------|
| 🏠 Home | Project dashboard with dataset overview and model summary |
| 📊 Dataset Explorer | Filterable table, statistics, class distribution, missing-value analysis |
| 📈 Visualizations | Pair plot, histograms, correlation heatmap, box plots, 3D scatter |
| 🤖 Model Training | Train & compare Logistic Regression, Decision Tree, Random Forest, KNN, SVM |
| 🔮 Predict Species | Real-time prediction with probability bars and history export |
| 📋 Model Evaluation | Accuracy / F1 / Precision / Recall / Confusion Matrix for all models |
| ⭐ Feature Importance | Tree importances, LR coefficients, permutation importance |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd iris-classifier
pip install -r requirements.txt
```

### 2. Train models (done automatically on first launch)

```bash
python train_model.py
```

### 3. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
iris-classifier/
├── app.py                  # Main entry-point + global CSS + sidebar nav
├── train_model.py          # Data loading, training pipeline, model persistence
├── requirements.txt
├── iris_model.pkl          # Persisted models (auto-generated)
├── data/
│   └── IRIS.csv
└── pages/
    ├── Home.py             # Dashboard
    ├── Dataset.py          # Explorer
    ├── Visualization.py    # Charts
    ├── ModelTraining.py    # Training & comparison
    ├── Prediction.py       # Real-time prediction
    ├── Evaluation.py       # Metrics & confusion matrix
    └── FeatureImportance.py
```

---

## 🌐 Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy**

Add a `packages.txt` if any system dependencies are needed (none required here).

---

## 🎨 Tech Stack

- **Streamlit** — web framework
- **Scikit-Learn** — ML models, metrics, preprocessing
- **Pandas / NumPy** — data manipulation
- **Plotly** — interactive charts
- **Matplotlib / Seaborn** — static charts
- **Joblib** — model serialisation

---

## 📊 Models & Performance

| Model | Typical Accuracy |
|-------|-----------------|
| Logistic Regression | ~96–97% |
| Decision Tree | ~93–97% |
| Random Forest | ~96–97% |
| KNN | ~96–97% |
| SVM (RBF) | ~96–100% |

Results may vary slightly due to random splits; cross-validation scores are more stable.

---

## 📄 License

MIT — free to use, modify, and distribute.
