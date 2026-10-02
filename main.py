import json
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()  # must run before ml_engine is imported: it reads DATA_DIR at import time

from . import MLengine as ml_engine
from .MLengine import DataUnavailable, NotReady, extract_skills, normalize_skill, skill_engine

log = logging.getLogger("skillspring")
ROOT = Path(__file__).parent
_vercel = bool(os.getenv("VERCEL"))
database_url = os.getenv("DATABASE_URL") or (
    "sqlite:////tmp/skillspring.db" if _vercel else "sqlite:///./skillspring.db"
)
engine = create_engine(database_url, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


class Profile(Base):
    __tablename__ = "profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    skills_json: Mapped[str] = mapped_column(Text)
    predicted_role: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # MySQL may still be booting when Docker starts the web container.
    for attempt in range(15):
        try:
            Base.metadata.create_all(engine)
            break
        except Exception:
            if attempt == 14:
                raise
            time.sleep(2)
    # Vercel functions use an ephemeral, read-only deployment filesystem. The saved
    # model is bundled for serving there; background retraining stays local/Docker.
    if _vercel:
        if not ml_engine.MODEL_PATH.exists():
            log.error("The bundled model is missing: %s", ml_engine.MODEL_PATH)
        else:
            skill_engine.bootstrap()
    else:
        # Load the saved model, or train from the postings in ./data, without delaying startup.
        threading.Thread(target=skill_engine.bootstrap, daemon=True, name="ml-bootstrap").start()
    yield


app = FastAPI(title="SkillSpring", lifespan=lifespan)


class ProfileInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    skills: list[str] = Field(min_length=1, max_length=30)
    desired_role: str | None = Field(default=None, max_length=120)


def clean_skills(skills: list[str]) -> list[str]:
    return sorted({normalize_skill(s) for s in skills if s.strip()})


def run_analysis(skills: list[str], desired_role: str | None) -> dict:
    try:
        return skill_engine.analyse(skills, desired_role)
    except NotReady as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


def require_admin(token: str | None) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(403, "Set ADMIN_TOKEN in the environment to enable model administration.")
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(401, "Invalid admin token")


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/profiles")
def save_profile(payload: ProfileInput):
    skills = clean_skills(payload.skills)
    if not skills:
        raise HTTPException(400, "Enter at least one skill")
    result = run_analysis(skills, payload.desired_role)
    with Session(engine) as session:
        profile = Profile(name=payload.name.strip(), skills_json=json.dumps(skills),
                          predicted_role=result["predicted_role"])
        session.add(profile)
        session.commit()
        session.refresh(profile)
    return {"id": profile.id, **result}


@app.post("/api/resume")
async def analyse_resume(file: UploadFile = File(...), desired_role: str | None = Form(default=None)):
    # Deliberately not persisted: content lives only for this request.
    text = (await file.read()).decode("utf-8", errors="ignore").lower()
    skills = extract_skills(text)
    result = run_analysis(skills, desired_role)
    return {"detected_skills": skills, **result, "stored": False}


# ---- model administration (needs ADMIN_TOKEN, sent as the X-Admin-Token header)
@app.get("/api/model/status")
def model_status():
    return skill_engine.status()


@app.post("/api/model/train")
def train_now(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    try:
        return skill_engine.train()
    except RuntimeError as exc:  # includes DataUnavailable
        raise HTTPException(409, str(exc))


@app.post("/api/data/refresh")
def refresh_data(x_admin_token: str | None = Header(default=None)):
    """Pull fresh postings from Adzuna, then retrain on everything in ./data."""
    require_admin(x_admin_token)
    try:
        added = ml_engine.fetch_live_postings()
        return {"postings_added": added, "training": skill_engine.train()}
    except requests.RequestException as exc:
        raise HTTPException(502, f"Job API request failed: {exc}")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
