# 🏦 Credit Scoring System

> Système de scoring crédit performant, interprétable et équitable
> Développé par **Youssef Kassabi** — CMC Nouacer-Casablanca | 2025-2026

---

## 📋 Description

Ce projet implémente un système complet d'analyse du risque client pour l'octroi de crédit, couvrant :

- Prétraitement des données (WOE, StandardScaler)
- Modélisation par Régression Logistique
- Validation (AUC = 0.94, Gini = 0.87, KS = 0.71)
- Analyse de l'équité (Fairness < 2%)
- Dashboard interactif Power BI (5 pages)
- API de scoring FastAPI
- Interface web bancaire HTML/CSS

---

## 📊 Performance du Modèle

| Métrique | Valeur | Seuil Bâle III |
|---|---|---|
| AUC-ROC | **0.94** | ≥ 0.75 ✅ |
| Gini | **0.87** | ≥ 0.50 ✅ |
| KS Statistic | **0.71** | — ✅ |
| Accuracy | **91%** | — ✅ |
| Precision | **90%** | — ✅ |
| Recall | **70%** | — ⚠️ |
| Écart Fairness | **2.09%** | < 20% ✅ |

---

## 🗂️ Structure du Projet

credit-scoring-system/
│
├── 📓 notebook/
│   └── credit_scoring.ipynb
│
├── 🤖 api/
│   ├── main.py
│   ├── credit_model.pkl
│   └── scaler.pkl
│
├── 🌐 interface/
│   └── index.html
│
├── 📊 dashboard/
│   └── credit_scoring.pbix
│
└── 📄 README.md

---

## 🚀 Lancer l'API

pip install fastapi uvicorn pydantic joblib scikit-learn pandas

python main.py

API : http://127.0.0.1:8080
Swagger : http://127.0.0.1:8080/docs

---

## 📡 Exemple de Réponse

{
  "probabilite_defaut": 0.0046,
  "credit_score": 868.45,
  "decision": "Accepté ✅"
}

---

## 📈 Dashboard Power BI

| Page | Contenu |
|---|---|
| 1 | Performance & Risque du Portefeuille |
| 2 | Analyse Opérationnelle & Équité |
| 3 | Simulation Économique & Stratégie d'Octroi |
| 4 | Équité & Fairness du Modèle |
| 5 | Outil d'Aide à la Décision |

---

## 🛠️ Technologies

| Catégorie | Technologies |
|---|---|
| Modélisation | Python, Scikit-learn, Logistic Regression |
| Validation | AUC-ROC, Gini, KS, Matrice de Confusion |
| Fairness | Écart d'approbation, Demographic Parity |
| Reporting | Power BI, DAX |
| Déploiement | FastAPI, Uvicorn |
| Interface | HTML, CSS, JavaScript |
| Versioning | GitHub |

---

## 👤 Auteur  :Youssef Kassabi
 : intelligence artificielle option DATA ANALYST(OFPPT)
Filière : Intelligence Artificielle — Assistant Data Analyst
CMC Nouacer-Casablanca | 2025-2026
Encadrant : M. Saber Hamza
