# SkillSpring

SkillSpring predicts the skills a user needs to get placed in their desired role.
It trains on **real job-posting data** (no synthetic/hardcoded skill lists) and
recommends what to learn next, backed by nine machine learning techniques.

## What it does

1. A user submits their current skills (via a form or a resume upload) and,
   optionally, a desired role.
2. The app predicts their best-fit role from their skills, and shows:
   - which skills they already match for that role
   - a prioritized list of skills to learn next, with reasons
   - a "readiness" score
   - trending / rising-demand skills
   - a short-term demand forecast for the role

## How the ML works

The project is trained entirely on real job postings (a Kaggle job-postings
dataset, plus live postings pulled from job-search APIs). No skill list or
demand score is hardcoded.

| Algorithm | Role in the pipeline |
|---|---|
| **Isolation Forest** | Removes anomalous / junk postings before training |
| **Random Forest (classifier)** | Predicts the best-fit role from a skill list |
| **Apriori** + **FP-Growth** | Mine skill co-occurrence rules (e.g. "Python + SQL → Pandas") |
| **K-Means** | Clusters postings into skill families, for cluster-based recommendations |
| **Moving Average** | Smoothed monthly demand per (role, skill) |
| **Growth Rate** | Change in demand: last 3 months vs. the previous 3 months |
| **Linear Regression**, **Random Forest Regression**, **XGBoost Regression** | Forecast next-month demand per skill; the best of these (or the moving average, if none beats it) is used for serving |

The final "what to learn next" ranking blends role demand, rule strength,
forecast demand, and cluster affinity.

## Project structure

```
.
├── main.py           # FastAPI app: routes, profile storage, admin endpoints
├── ml_engine.py       # Data loading, live-API fetchers, training, and serving logic
├── requirements.txt
├── .env.example       # Template for required environment variables
├── .gitignore
├── static/            # Frontend (index.html, served at "/")
└── data/               # Job-posting CSVs go here (not committed to git)
    ├── job_postings.csv        # e.g. a Kaggle job-postings export
    ├── postings_live.csv       # auto-generated: live Adzuna postings
    ├── postings_jooble.csv     # auto-generated: live Jooble postings
    └── model.joblib            # auto-generated: the trained model
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add a training dataset
Download a job-postings dataset (e.g. the LinkedIn Job Postings dataset on
Kaggle) and place the postings CSV in `data/`. It needs at minimum a job-title
column; a description/skills column and a date column are used when present
and improve results (the date column enables trend and forecast features).

### 3. Configure environment variables
Copy the example file and fill in your own values:
```bash
cp .env.example .env
```
Then edit `.env` and set:
- `ADMIN_TOKEN` — protects the training/refresh endpoints; pick your own
  long random string, it isn't provided by any service.
- `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` — optional, from developer.adzuna.com.
- `JOOBLE_API_KEY` — optional, from Jooble's API page.

Live-source variables can be left blank if you're not using that source.
**Never commit your real `.env` file** — it's already excluded in
`.gitignore`.

### 4. Run the app
```bash
uvicorn main:app --reload
```

On startup, the app trains (or loads a previously saved model) in the
background from whatever CSVs are in `data/`. Check progress at:

```
GET /api/model/status
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/api/profiles` | POST | Submit `{name, skills[], desired_role?}`, get role prediction + upskilling plan |
| `/api/resume` | POST | Upload a resume file (multipart `file`, optional `desired_role` form field); skills are extracted automatically |
| `/api/model/status` | GET | Training status, report, and diagnostics |
| `/api/model/train` | POST | Retrain on the current contents of `data/` (requires `X-Admin-Token` header) |
| `/api/data/refresh` | POST | Fetch fresh postings from all configured live sources, then retrain (requires `X-Admin-Token` header) |

### Example
```bash
curl -X POST http://localhost:8000/api/profiles \
  -H "Content-Type: application/json" \
  -d '{"name":"Hari","skills":["python","sql"],"desired_role":"Data Scientist"}'
```

## Keeping data fresh

`/api/data/refresh` pulls new postings from every configured live source
(Adzuna, Jooble) and retrains. Since trend and forecast features need
several months of dated history, schedule this to run regularly, e.g. with
a weekly cron job:

```bash
0 2 * * 1 curl -s -X POST http://localhost:8000/api/data/refresh -H "X-Admin-Token: $ADMIN_TOKEN"
```

## Notes

- If dated postings aren't available (or don't cover enough months), trend
  and forecast outputs are disabled automatically; role prediction, skill
  gaps, co-occurrence rules and clustering are unaffected.
- Role and skill definitions are pattern/vocabulary based
  (`ROLE_TITLE_PATTERNS`, `SKILL_VOCAB` in `ml_engine.py`) and can be edited
  or expanded to cover more roles, skills, or domains.
