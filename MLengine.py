
from __future__ import annotations

import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

log = logging.getLogger("skillspring.ml")
os.environ.setdefault("JOOBLE_API_KEY", "dummy")  # for Kaggle job-postings export

ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_PATH = DATA_DIR / "model.joblib"
LIVE_PATH = DATA_DIR / "postings_live.csv"

# ---------------------------------------------------------------- tunables
RANDOM_STATE = 42
MIN_POSTINGS = 200            # refuse to train on less usable data than this
MIN_ROLE_POSTINGS = 40        # drop roles with fewer postings
MIN_SKILLS_PER_POSTING = 2    # postings with fewer recognised skills are noise
MIN_SKILL_SUPPORT = 0.005     # skill must appear in >=0.5% of postings (and >=10)
MIN_DEMAND = 0.05             # a skill is "expected" for a role if in >=5% of its postings
CONTAMINATION = 0.02          # share of postings Isolation Forest may discard
RULE_MIN_SUPPORT = 0.03
RULE_MIN_CONFIDENCE = 0.5
RULE_MIN_LIFT = 1.1
MAX_RULES = 5000
MIN_MONTH_POSTINGS = 15       # months with fewer postings for a role are ignored
MIN_MONTHS = 6                # months of history needed for MA / growth rate
FEATURES = ["lag1", "lag2", "lag3", "ma3", "growth"]
WEIGHTS = {"demand": 0.50, "rules": 0.20, "forecast": 0.15, "cluster": 0.15}


class DataUnavailable(RuntimeError):
    """No (or not enough) real data to train on."""


class NotReady(RuntimeError):
    """The model has not been trained yet."""


# ---------------------------------------------------------------- skill vocabulary
# This is only the *dictionary of skill names* used to read postings and resumes.
# It is not training data: which skills each role needs is learned from postings.
SKILL_VOCAB: dict[str, list[str]] = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript"], "c++": ["c++"], "c#": ["c#"], "golang": ["golang"],
    "rust": ["rust"], "php": ["php"], "ruby": ["ruby", "ruby on rails"], "kotlin": ["kotlin"],
    "swift": ["swift"], "scala": ["scala"], "matlab": ["matlab"],
    "bash": ["bash", "shell scripting"], "sql": ["sql"], "html": ["html", "html5"],
    "css": ["css", "css3"],
    "react": ["react", "reactjs", "react.js"], "angular": ["angular", "angularjs"],
    "vue": ["vue", "vue.js", "vuejs"], "next.js": ["next.js", "nextjs"],
    "node.js": ["node.js", "nodejs"], "express": ["express.js", "expressjs"],
    "django": ["django"], "flask": ["flask"], "fastapi": ["fastapi"],
    "spring": ["spring boot", "spring framework"], ".net": [".net", "dotnet", "asp.net"],
    "api": ["api", "apis", "rest api", "rest apis", "restful", "restful apis"],
    "graphql": ["graphql"], "microservices": ["microservices", "microservice"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"], "computer vision": ["computer vision"],
    "llm": ["llm", "llms", "large language models"],
    "generative ai": ["generative ai", "genai"], "mlops": ["mlops"],
    "pandas": ["pandas"], "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "tensorflow": ["tensorflow"], "pytorch": ["pytorch"], "keras": ["keras"],
    "xgboost": ["xgboost"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling"],
    "data analysis": ["data analysis", "data analytics"],
    "data visualization": ["data visualization", "data visualisation"],
    "tableau": ["tableau"], "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "advanced excel"], "spark": ["apache spark", "pyspark"],
    "hadoop": ["hadoop"], "kafka": ["kafka"], "airflow": ["airflow"], "etl": ["etl", "elt"],
    "data warehousing": ["data warehouse", "data warehousing"], "snowflake": ["snowflake"],
    "dbt": ["dbt"], "a/b testing": ["a/b testing", "ab testing"],
    "mysql": ["mysql"], "postgresql": ["postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo db"], "redis": ["redis"], "nosql": ["nosql"],
    "elasticsearch": ["elasticsearch"], "dynamodb": ["dynamodb"],
    "aws": ["aws", "amazon web services"], "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "terraform": ["terraform"],
    "ansible": ["ansible"], "jenkins": ["jenkins"],
    "ci/cd": ["ci/cd", "cicd", "continuous integration"],
    "git": ["git", "github", "gitlab"], "linux": ["linux", "unix"],
    "prometheus": ["prometheus"], "grafana": ["grafana"], "serverless": ["serverless"],
    "networking": ["networking", "tcp/ip"],
    "security": ["security", "cybersecurity", "cyber security", "information security", "infosec"],
    "siem": ["siem", "splunk"], "incident response": ["incident response"],
    "penetration testing": ["penetration testing", "pentesting", "pen testing"],
    "vulnerability management": ["vulnerability management", "vulnerability assessment"],
    "firewalls": ["firewall", "firewalls"],
    "iam": ["iam", "identity and access management"],
    "encryption": ["encryption", "cryptography"], "owasp": ["owasp"],
    "unit testing": ["unit testing", "unit tests", "pytest", "junit", "jest"],
    "selenium": ["selenium"], "agile": ["agile", "scrum"], "jira": ["jira"],
    "android": ["android"], "ios": ["ios"], "react native": ["react native"],
    "flutter": ["flutter"],
}
ALIAS_TO_CANON = {form: canon for canon, forms in SKILL_VOCAB.items() for form in forms}
_SKILL_RE = re.compile(
    r"(?<![a-z0-9+#.])("
    + "|".join(re.escape(f) for f in sorted(ALIAS_TO_CANON, key=len, reverse=True))
    + r")(?![a-z0-9+#])",
    re.I,
)


def extract_skills(text: str) -> list[str]:
    """Canonical skill names found in free text (postings, resumes)."""
    return sorted({ALIAS_TO_CANON[m.lower()] for m in _SKILL_RE.findall(text or "")})


def normalize_skill(skill: str) -> str:
    s = skill.strip().lower()
    return ALIAS_TO_CANON.get(s, s)


# Job title -> canonical role. First match wins. Roles are learned only if the data has
# at least MIN_ROLE_POSTINGS postings for them.
ROLE_TITLE_PATTERNS = [
    ("Full Stack Developer", r"full[\s-]?stack"),
    ("Frontend Developer", r"front[\s-]?end|ui developer|react developer|angular developer|web developer"),
    ("Backend Developer", r"back[\s-]?end|api developer|python developer|java developer|\.net developer"),
    ("Machine Learning Engineer", r"machine learning engineer|\bml engineer|\bai engineer|deep learning"),
    ("Data Scientist", r"data scientist|research scientist|applied scientist"),
    ("Data Analyst", r"data analyst|business analyst|\bbi analyst|analytics"),
    ("Data Engineer", r"data engineer|\betl\b|big data"),
    ("Cloud Engineer", r"cloud|solutions architect|site reliability|\bsre\b|platform engineer"),
    ("DevOps Engineer", r"devops|release engineer"),
    ("Cybersecurity Analyst", r"security|cyber|soc analyst|penetration|infosec"),
]
_ROLE_RES = [(role, re.compile(pat, re.I)) for role, pat in ROLE_TITLE_PATTERNS]
ADZUNA_SEARCH_TERMS = [
    "data scientist", "data analyst", "data engineer", "machine learning engineer",
    "backend developer", "frontend developer", "full stack developer", "cloud engineer",
    "devops engineer", "cybersecurity analyst",
]


def map_role(title: str) -> str | None:
    for role, rx in _ROLE_RES:
        if rx.search(title or ""):
            return role
    return None


# ---------------------------------------------------------------- real data loading
TITLE_COLS = ["title", "job_title", "job title", "jobtitle", "position", "name"]
DESC_COLS = ["description", "job_description", "job description", "jobdescription", "details"]
SKILL_COLS = ["skills", "job_skills", "skill", "required_skills", "key skills", "skills_desc"]
DATE_COLS = ["date", "posted_date", "date_posted", "created", "listed_time",
             "original_listed_time", "posted", "first_seen", "date_added"]


def _pick(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((n for n in names if n in df.columns), None)


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        vals = series.dropna()
        if vals.empty:
            return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        unit = "ms" if vals.median() > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    return parsed.dt.tz_localize(None)


def load_postings() -> pd.DataFrame:
    """Read every CSV in DATA_DIR (e.g. a Kaggle job-postings export plus the live file).

    Column names are matched flexibly; a title column is required, description / skills /
    date columns are used when present.
    """
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        try:
            raw = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
        except Exception as exc:  # unreadable file should not kill training
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        title_c = _pick(raw, TITLE_COLS)
        if title_c is None:
            log.warning("Skipping %s: no title column (looked for %s)", path.name, TITLE_COLS)
            continue
        body_cols = list(dict.fromkeys(c for c in (_pick(raw, DESC_COLS), _pick(raw, SKILL_COLS)) if c))
        titles = raw[title_c].fillna("").astype(str)
        # Skills are read from description/skills columns, not the title, so the role
        # label is not leaked into the features. Fall back to the title if no body.
        body = raw[body_cols].fillna("").astype(str).agg(" ".join, axis=1) if body_cols else titles
        date_c = _pick(raw, DATE_COLS)
        frames.append(pd.DataFrame({
            "title": titles,
            "text": body.str.lower(),
            "posted": _parse_dates(raw[date_c]) if date_c else pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]"),
            "source": path.name,
        }))
    if not frames:
        raise DataUnavailable(
            f"No job-posting CSVs found in {DATA_DIR}. Download a job-postings dataset "
            "(e.g. Kaggle LinkedIn/Naukri postings) into that folder, or set ADZUNA_APP_ID / "
            "ADZUNA_APP_KEY and call POST /api/data/refresh."
        )
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)


def fetch_live_postings(pages: int = 5, per_page: int = 50) -> int:
    """Pull fresh postings from Adzuna and append them to LIVE_PATH. Returns rows added.

    Adzuna returns a truncated description snippet, and only current listings, so
    month-over-month trends need history to accumulate: schedule this regularly.
    """
    app_id, app_key = os.getenv("ADZUNA_APP_ID"), os.getenv("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        raise DataUnavailable("Set ADZUNA_APP_ID and ADZUNA_APP_KEY (free keys from developer.adzuna.com).")
    country = os.getenv("ADZUNA_COUNTRY", "in")
    rows = []
    for term in ADZUNA_SEARCH_TERMS:
        for page in range(1, pages + 1):
            resp = requests.get(
                f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}",
                params={"app_id": app_id, "app_key": app_key, "results_per_page": per_page,
                        "what": term, "content-type": "application/json"},
                timeout=20,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                break
            rows += [{"id": j.get("id"), "title": j.get("title"), "description": j.get("description"),
                      "created": j.get("created")} for j in results]
            time.sleep(0.3)  # stay polite to the API
    fresh = pd.DataFrame(rows)
    if fresh.empty:
        return 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    before = 0
    if LIVE_PATH.exists():
        old = pd.read_csv(LIVE_PATH, dtype={"id": str})
        before = len(old)
        fresh = pd.concat([old, fresh.astype({"id": str})], ignore_index=True)
    fresh = fresh.astype({"id": str}).drop_duplicates(subset="id")
    fresh.to_csv(LIVE_PATH, index=False)
    return len(fresh) - before


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["role"] = df["title"].map(map_role)
    df = df.dropna(subset=["role"]).copy()
    df["skills"] = df["text"].map(extract_skills)
    df = df[df["skills"].map(len) >= MIN_SKILLS_PER_POSTING]
    counts = df["role"].value_counts()
    df = df[df["role"].isin(counts[counts >= MIN_ROLE_POSTINGS].index)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- training pieces
def _scores(y, pred) -> dict:
    return {"rmse": round(float(mean_squared_error(y, pred) ** 0.5), 5),
            "mae": round(float(mean_absolute_error(y, pred)), 5),
            "r2": round(float(r2_score(y, pred)), 4)}


def _regressors() -> dict:
    # XGBoost is used only during model training. Import it lazily so the
    # production API does not load this large training-only dependency.
    from xgboost import XGBRegressor

    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                               n_jobs=-1, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                                colsample_bytree=0.9, n_jobs=-1, random_state=RANDOM_STATE),
    }


def _mine_rules(xbool: pd.DataFrame, report: dict) -> list[tuple]:
    """Run Apriori and FP-Growth. They must find identical itemsets; that is verified on a
    sample, then the configured miner (RULE_MINER, default fpgrowth) runs on all data."""
    # Training-only dependency: do not load mlxtend when the deployed API only
    # needs to load the already-trained model.
    from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
    sample = xbool.sample(min(len(xbool), 20000), random_state=RANDOM_STATE)
    kw = dict(min_support=RULE_MIN_SUPPORT, use_colnames=True, max_len=4)
    fp_sample, ap_sample = fpgrowth(sample, **kw), apriori(sample, **kw)
    miner_name = os.getenv("RULE_MINER", "fpgrowth").lower()
    miner = apriori if miner_name == "apriori" else fpgrowth
    freq = miner(xbool, **kw)
    rules: list[tuple] = []
    if len(freq) and freq["itemsets"].map(len).max() >= 2:
        df = association_rules(freq, num_itemsets=len(xbool), metric="confidence",
                               min_threshold=RULE_MIN_CONFIDENCE)
        df = df[(df["lift"] >= RULE_MIN_LIFT) & (df["consequents"].map(len) == 1)]
        df = df.assign(score=df["confidence"] * df["lift"]).nlargest(MAX_RULES, "score")
        rules = [(tuple(sorted(a)), next(iter(c)), float(cf), float(lf), float(sp))
                 for a, c, cf, lf, sp in zip(df["antecedents"], df["consequents"], df["confidence"],
                                            df["lift"], df["support"])]
    report["rules"] = {
        "miner_used": miner_name, "frequent_itemsets": int(len(freq)), "rules_kept": len(rules),
        "apriori_fpgrowth_agree_on_sample": bool(set(fp_sample["itemsets"]) == set(ap_sample["itemsets"])),
        "min_support": RULE_MIN_SUPPORT, "min_confidence": RULE_MIN_CONFIDENCE,
    }
    return rules


def _trend_models(posts: pd.DataFrame, xdf: pd.DataFrame, report: dict) -> dict | None:
    """Monthly demand panels -> Moving Average, Growth Rate and the three regressors."""
    dated = posts["posted"].notna().to_numpy()
    if dated.sum() < MIN_POSTINGS:
        report["trends"] = {"enabled": False, "reason": "too few postings carry a date"}
        return None
    d = xdf.loc[dated].reset_index(drop=True)
    role_s = posts.loc[dated, "role"].reset_index(drop=True).rename("role")
    month_s = posts.loc[dated, "posted"].dt.to_period("M").dt.to_timestamp().reset_index(drop=True).rename("month")
    counts = d.groupby([role_s, month_s]).size()
    shares = d.groupby([role_s, month_s]).mean()
    month_total = d.groupby(month_s).size()

    panels, volume_growth = {}, {}
    for role in sorted(role_s.unique()):
        c = counts.xs(role, level="role")
        ok = c[c >= MIN_MONTH_POSTINGS]
        if len(ok) < MIN_MONTHS:
            continue
        full = pd.date_range(ok.index.min(), ok.index.max(), freq="MS")
        panels[role] = shares.xs(role, level="role").loc[ok.index].reindex(full)
        # role's share of ALL postings that month: robust to a partially-filled latest month
        role_share = (c / month_total.reindex(c.index)).reindex(full)
        recent, prior = role_share.iloc[-3:].mean(), role_share.iloc[-6:-3].mean()
        if pd.notna(recent) and pd.notna(prior) and prior > 0:
            volume_growth[role] = float((recent - prior) / prior)
    if not panels:
        report["trends"] = {"enabled": False,
                            "reason": f"no role has {MIN_MONTHS}+ months with {MIN_MONTH_POSTINGS}+ postings"}
        return None

    growth, recent_ma, forecast = {}, {}, {}
    for role, W in panels.items():
        recent, prior = W.iloc[-3:].mean(), W.iloc[-6:-3].mean()          # Moving Average (3 mo)
        recent_ma[role] = recent
        growth[role] = ((recent - prior) / prior.clip(lower=0.01)).fillna(0.0)  # Growth Rate

    # ---- supervised set: predict a skill's share next month from its own history
    rows = []
    for role, W in panels.items():
        l1, l2, l3 = W.shift(1), W.shift(2), W.shift(3)
        feats = {"lag1": l1, "lag2": l2, "lag3": l3, "ma3": (l1 + l2 + l3) / 3,
                 "growth": ((l1 - l3) / l3.clip(lower=0.01)).clip(-1, 3)}

        def long(F: pd.DataFrame, name: str) -> pd.DataFrame:
            return F.rename_axis("month").reset_index().melt(id_vars="month", var_name="skill", value_name=name)

        frame = long(W, "target")
        for name, F in feats.items():
            frame[name] = long(F, name)[name].to_numpy()
        frame["role"] = role
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True).dropna()
    months = np.sort(data["month"].unique())
    trend_report: dict = {"enabled": True, "months_of_history": int(max(len(p) for p in panels.values())),
                          "roles_with_trends": sorted(panels), "regression": None}
    report["trends"] = trend_report

    best_name, final_model = "moving_average", None
    if len(months) >= 4 and len(data) >= 300:
        cut = months[max(1, int(len(months) * 0.8))]      # time-ordered split: no peeking ahead
        train_df, test_df = data[data["month"] < cut], data[data["month"] >= cut]
        if len(train_df) >= 100 and len(test_df) >= 50:
            scores = {"moving_average": _scores(test_df["target"], test_df["ma3"])}
            for name, model in _regressors().items():
                model.fit(train_df[FEATURES], train_df["target"])
                scores[name] = _scores(test_df["target"], model.predict(test_df[FEATURES]))
            best_name = min(scores, key=lambda k: scores[k]["rmse"])
            if best_name != "moving_average":
                final_model = _regressors()[best_name].fit(data[FEATURES], data["target"])
            trend_report["regression"] = {"holdout_months": int((months >= cut).sum()),
                                          "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
                                          "scores": scores, "selected": best_name}
    if trend_report["regression"] is None:
        trend_report["regression"] = {"skipped": "not enough monthly history for a hold-out split",
                                      "selected": "moving_average"}

    for role, W in panels.items():                       # forecast next month
        l1, l2, l3 = W.iloc[-1], W.iloc[-2], W.iloc[-3]
        f = pd.DataFrame({"lag1": l1, "lag2": l2, "lag3": l3})
        f["ma3"] = (f["lag1"] + f["lag2"] + f["lag3"]) / 3
        f["growth"] = ((f["lag1"] - f["lag3"]) / f["lag3"].clip(lower=0.01)).clip(-1, 3)
        f = f.dropna()
        pred = f["ma3"] if final_model is None else pd.Series(final_model.predict(f[FEATURES]), index=f.index)
        forecast[role] = pred.clip(0, 1)

    starts = [p.index.min() for p in panels.values()]
    ends = [p.index.max() for p in panels.values()]
    return {"growth": growth, "recent": recent_ma, "forecast": forecast, "volume_growth": volume_growth,
            "window": (min(starts), max(ends))}


def _role_summaries(posts: pd.DataFrame, trends: dict | None) -> dict:
    counts, total = posts["role"].value_counts(), len(posts)
    max_share = counts.max() / total
    window = "n/a"
    if trends:
        a, b = trends["window"]
        window = f"{a:%b %Y} – {b:%b %Y}"
    out = {}
    for role, n in counts.items():
        share = n / total
        vg = trends["volume_growth"].get(role) if trends else None
        if vg is None:
            outlook = "Trend unavailable (not enough dated postings)"
            score = round(100 * share / max_share)
            reason = f"{n:,} postings analysed ({share:.0%} of the dataset)."
        else:
            outlook = ("Strong growth signal" if vg > 0.25 else "Steady growth signal" if vg > 0.05
                       else "Stable demand signal" if vg > -0.05 else "Cooling demand signal")
            score = round(100 * (0.5 * share / max_share + 0.5 * float(np.clip(0.5 + vg / 2, 0, 1))))
            reason = (f"{n:,} postings analysed; the role's share of monthly postings changed "
                      f"{vg:+.0%} versus the previous 3 months.")
        out[role] = {"outlook": outlook, "years": window, "score": int(score), "reason": reason,
                     "postings": int(n), "volume_growth": None if vg is None else round(vg, 3)}
    return out


def train_model() -> dict:
    """Train every model on the real postings in DATA_DIR and return the artifact."""
    t0 = time.time()
    raw = load_postings()
    posts = prepare(raw)
    if len(posts) < MIN_POSTINGS:
        raise DataUnavailable(
            f"Only {len(posts)} usable postings found (need {MIN_POSTINGS}+). Usable = the title maps "
            f"to a known role and the text mentions at least {MIN_SKILLS_PER_POSTING} known skills."
        )
    report: dict = {"raw_postings": int(len(raw)), "usable_postings": int(len(posts))}

    mlb = MultiLabelBinarizer(classes=sorted(SKILL_VOCAB))
    x_all = mlb.fit_transform(posts["skills"]).astype(np.uint8)
    keep = x_all.mean(axis=0) >= max(MIN_SKILL_SUPPORT, 10 / len(posts))
    vocab = [s for s, k in zip(mlb.classes_, keep) if k]
    x = x_all[:, keep]

    # 1) Isolation Forest: discard anomalous postings
    iso_feats = np.column_stack([x, x.sum(axis=1)])
    inlier = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=RANDOM_STATE,
                             n_jobs=-1).fit(iso_feats).predict(iso_feats) == 1
    report["isolation_forest"] = {"removed": int((~inlier).sum()), "contamination": CONTAMINATION}
    posts, x = posts[inlier].reset_index(drop=True), x[inlier]
    y = posts["role"].to_numpy()
    roles = sorted(set(y))
    report["role_counts"] = {r: int((y == r).sum()) for r in roles}
    report["date_range"] = (None if posts["posted"].notna().sum() == 0 else
                            [str(posts["posted"].min().date()), str(posts["posted"].max().date())])

    # 2) Random Forest classifier: skills -> role. Users have partial skill lists while postings
    #    list many skills, so training also sees randomly thinned copies (skill dropout).
    rng = np.random.default_rng(RANDOM_STATE)

    def thin(a: np.ndarray) -> np.ndarray:
        return (a * (rng.random(a.shape) < 0.5)).astype(np.uint8)

    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    def new_clf() -> RandomForestClassifier:
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=RANDOM_STATE)

    probe = new_clf().fit(np.vstack([x_tr, thin(x_tr)]), np.concatenate([y_tr, y_tr]))
    x_te_thin = thin(x_te)
    report["role_classifier"] = {
        "accuracy_full_skill_list": round(float(accuracy_score(y_te, probe.predict(x_te))), 3),
        "accuracy_partial_skill_list": round(float(accuracy_score(y_te, probe.predict(x_te_thin))), 3),
        "macro_f1_partial_skill_list": round(float(f1_score(y_te, probe.predict(x_te_thin), average="macro")), 3),
        "test_rows": int(len(y_te)),
    }
    clf = new_clf().fit(np.vstack([x, thin(x)]), np.concatenate([y, y]))

    xdf = pd.DataFrame(x, columns=vocab)
    demand = xdf.groupby(y).mean()                       # role x skill: share of postings

    # 3) Apriori + FP-Growth: skill co-occurrence rules
    rules = _mine_rules(xdf.astype(bool), report)

    # 4) K-Means: skill families (k picked by silhouette)
    sample = rng.choice(len(x), size=min(len(x), 6000), replace=False)
    best = None
    for k in range(3, min(10, len(vocab), len(x) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=RANDOM_STATE).fit(x)
        try:
            sil = float(silhouette_score(x[sample], km.labels_[sample]))
        except ValueError:
            continue
        if best is None or sil > best[0]:
            best = (sil, k, km)
    if best is None:
        raise DataUnavailable("K-Means could not find a valid clustering on this data.")
    sil, k, kmeans = best
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=vocab)
    cluster_names = [" · ".join(centroids.iloc[i].nlargest(3).index) for i in range(k)]
    report["kmeans"] = {"k": int(k), "silhouette": round(sil, 3), "clusters": cluster_names}

    # 5) Moving Average, Growth Rate, Linear / Random Forest / XGBoost regression
    trends = _trend_models(posts, xdf, report)

    return {
        "version": 1, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vocab": vocab, "roles": roles, "classifier": clf, "kmeans": kmeans, "centroids": centroids,
        "cluster_names": cluster_names, "demand": demand, "rules": rules,
        "growth": trends["growth"] if trends else {}, "recent": trends["recent"] if trends else {},
        "forecast": trends["forecast"] if trends else {},
        "role_summary": _role_summaries(posts, trends),
        "report": {**report, "train_seconds": round(time.time() - t0, 1)},
    }


# ---------------------------------------------------------------- serving
class SkillEngine:
    """Thread-safe holder for the trained artifact; serves predictions."""

    def __init__(self) -> None:
        self._art: dict | None = None
        self._lock = threading.Lock()
        self.training = False
        self.error: str | None = None

    def bootstrap(self) -> None:
        """Load the saved model, or train from whatever data exists. Never raises."""
        try:
            if MODEL_PATH.exists():
                self._art = joblib.load(MODEL_PATH)
                return
        except Exception as exc:
            log.warning("Saved model unreadable (%s); retraining", exc)
        try:
            self.train()
        except Exception as exc:
            log.warning("Model not trained at startup: %s", exc)

    def train(self) -> dict:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Training is already running.")
        self.training = True
        try:
            art = train_model()
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            tmp = MODEL_PATH.with_suffix(".tmp")
            joblib.dump(art, tmp)
            tmp.replace(MODEL_PATH)
            self._art, self.error = art, None
            return art["report"]
        except Exception as exc:
            self.error = str(exc)
            raise
        finally:
            self.training = False
            self._lock.release()

    def status(self) -> dict:
        a = self._art
        return {"ready": a is not None, "training": self.training, "error": self.error,
                "trained_at": a["trained_at"] if a else None, "roles": a["roles"] if a else [],
                "report": a["report"] if a else None}

    def analyse(self, skills: list[str], desired_role: str | None = None, top_n: int = 8) -> dict:
        a = self._art
        if a is None:
            raise NotReady(self.error or "The model is still training or has no data yet. "
                                          "Add job-posting data and train it (see /api/model/status).")
        vocab = a["vocab"]
        index = {s: i for i, s in enumerate(vocab)}
        held = {s for s in skills if s in index}
        unrecognised = sorted(set(skills) - held)
        vec = np.zeros((1, len(vocab)), dtype=np.uint8)
        for s in held:
            vec[0, index[s]] = 1

        clf = a["classifier"]
        proba = clf.predict_proba(vec)[0]
        order = np.argsort(proba)[::-1]
        predicted = str(clf.classes_[order[0]])
        role_probs = [{"role": str(clf.classes_[i]), "probability": round(float(proba[i]), 3)} for i in order[:3]]

        target = predicted
        if desired_role:
            match = {r.lower(): r for r in a["roles"]}.get(desired_role.strip().lower())
            if match is None:
                raise ValueError(f"Unknown role '{desired_role}'. Available roles: {', '.join(a['roles'])}")
            target = match

        demand = a["demand"].loc[target]
        role_top = demand[demand >= MIN_DEMAND].sort_values(ascending=False)
        cand = [s for s in role_top.index if s not in held]

        rule_best: dict[str, tuple] = {}
        cand_set = set(cand)
        for ant, cons, conf, lift, _ in a["rules"]:
            if cons in cand_set and held.issuperset(ant):
                sc = conf * lift
                if sc > rule_best.get(cons, (0.0,))[0]:
                    rule_best[cons] = (sc, ant, lift)

        cluster = int(a["kmeans"].predict(vec)[0])
        cent = a["centroids"].iloc[cluster]
        fc, gr, rec = a["forecast"].get(target), a["growth"].get(target), a["recent"].get(target)

        comp = {"demand": np.array([demand[s] for s in cand]),
                "rules": np.array([rule_best.get(s, (0.0,))[0] for s in cand]),
                "cluster": np.array([cent[s] for s in cand]),
                "forecast": None if fc is None else np.array([fc.get(s, 0.0) for s in cand])}
        active = {k: w for k, w in WEIGHTS.items() if comp[k] is not None and len(cand) and comp[k].max() > 0}
        total_w = sum(active.values()) or 1.0
        priority = (sum(w / total_w * comp[k] / comp[k].max() for k, w in active.items())
                    if cand else np.array([]))

        upskill = []
        for i in np.argsort(-priority)[:top_n]:
            s = cand[i]
            why = [f"Appears in {demand[s]:.0%} of {target} postings"]
            if s in rule_best:
                _, ant, lift = rule_best[s]
                why.append(f"Usually paired with {', '.join(ant)} (lift {lift:.1f}x)")
            g = None if gr is None else float(gr.get(s, 0.0))
            if g is not None and abs(g) >= 0.10:
                why.append(f"Demand {'rising' if g > 0 else 'falling'} {abs(g):.0%} over the last 3 months")
            upskill.append({
                "skill": s, "priority": int(round(float(priority[i]) * 100)),
                "demand_pct": round(float(demand[s]) * 100, 1),
                "growth_pct": None if g is None else round(g * 100, 1),
                "forecast_demand_pct": (round(float(fc[s]) * 100, 1) if fc is not None and s in fc.index else None),
                "reason": "; ".join(why),
            })

        top15 = role_top.head(15)
        readiness = int(round(100 * top15[[s for s in top15.index if s in held]].sum() / top15.sum())) if len(top15) else 0
        trending = []
        if gr is not None and rec is not None:
            g = gr[rec[rec >= MIN_DEMAND].index].sort_values(ascending=False)
            trending = [{"skill": s, "growth_pct": round(float(v) * 100, 1)} for s, v in g.head(5).items() if v > 0.10]

        return {
            "role": target, "predicted_role": predicted, "desired_role": desired_role or None,
            "role_probabilities": role_probs,
            "matched_skills": [s for s in role_top.index if s in held],
            "missing_skills": [u["skill"] for u in upskill[:5]],
            "upskill": upskill, "readiness": readiness, "skill_cluster": a["cluster_names"][cluster],
            "trending_skills": trending, "unrecognised_skills": unrecognised,
            "forecast": a["role_summary"][target],
        }


skill_engine = SkillEngine()