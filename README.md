# Credit Scoring & Portfolio Management Platform

Full-stack web application for automated credit risk assessment, built as a final graduation project (PFE). The platform combines a trained machine learning model with a banking-style interface to help credit analysts evaluate loan applications and monitor portfolio performance.

Developed as part of the *Assistant Data Analyst — Intelligence Artificielle* diploma at OFPPT (CMC Nouaceur, Casablanca-Settat), successfully defended.

## Context

Credit institutions need a fast, consistent, and explainable way to assess borrower risk. This project addresses that need by combining a scoring model with a role-based web application, allowing analysts and managers to review applications, track decisions, and export reports without relying on manual spreadsheets.

## Features

- Role-based authentication (Admin, Manager, Analyst), each with its own permissions
- Real-time credit scoring, displayed on a 300–850 gauge
- Portfolio dashboards built with Chart.js
- Interface available in French, English, and Arabic
- Dark and light themes
- Excel export of reports (SheetJS)
- Print-ready loan decision summaries
- Fairness checks on the underlying model (Fairlearn)

## Model

The scoring model is a Logistic Regression trained on the LendingClub dataset (Kaggle, 3,816 records). Logistic Regression was chosen over more complex models such as XGBoost primarily for interpretability — under Bâle III requirements, every score needs to be explainable to auditors and regulators, which a linear model provides more directly than a black-box one.

Performance on the test set:

| Metric | Value |
|---|---|
| AUC | 0.94 |
| Gini | 0.87 |
| KS statistic | 0.71 |
| Accuracy | 91% |

Feature selection was done using Information Value (IV) via `scorecardpy`. The final probability is converted to a score using:

```
Score = 600 + 50 × log((1 − p) / p)
```

where `p` is the model's predicted probability of default.

## Architecture

The frontend is a static HTML/CSS/JS page (`credit_scoring_final.html`) that communicates with a FastAPI backend. The backend handles authentication, business logic, and serves predictions from the serialized model (`credit_model.pkl`, `scaler.pkl`), and persists data through SQLAlchemy to a PostgreSQL database (`credit_scoring`).

```
Frontend (HTML/CSS/JS)  →  FastAPI backend  →  PostgreSQL (credit_scoring)
                                 ↓
                     ML model (credit_model.pkl + scaler.pkl)
```

## Setup

Requirements: Python 3.10+, PostgreSQL 14+

```bash
git clone https://github.com/kbyussef/Credit-scoring-system.git
cd Credit-scoring-system
pip install -r requirements.txt
```

Create a `.env` file with the database connection string. If the database password contains an `@` character, it must be URL-encoded as `%40`:

```
DATABASE_URL=postgresql://user:password%40@localhost:5432/credit_scoring
```

Start the server:

```bash
uvicorn main:app --reload
```

On first run, seed the database:

```bash
curl -X POST http://localhost:8000/seed
```

The application is then available at `http://localhost:8000`.

## Reporting

A companion Power BI dashboard (5 pages) was built alongside the platform, with DAX measures such as `Revenu_Net_Estime`, `Perte_Estimee`, and `ROI_Portefeuille`, used to monitor portfolio-level indicators beyond individual scoring decisions.

## Tech stack

FastAPI, PostgreSQL, SQLAlchemy, scikit-learn, Chart.js, SheetJS, Power BI / DAX

## Author

Youssef Kassabi
[LinkedIn](https://www.linkedin.com/in/youssef-kassabi-367590372/)

## License

MIT — see [LICENSE](LICENSE).
