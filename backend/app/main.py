"""
DoseSense API.

Design notes that matter more than the endpoint list.

*The engine is built once at start-up.* Scoring six hundred patients across
their snapshot history takes a few seconds; doing it per request would make the
interface feel broken. The engine holds current assessments, probability
trajectories and evidence timelines in memory and serves them directly.

*The hidden adherence state is never served.* The ground-truth table is loaded
only by the evaluation path. The patient payload strips the generator's
archetype label before it leaves this process, so no interface can accidentally
display the answer next to the prediction. This is enforced by a test.

*Feedback is a first-class write.* A clinician marking an alert a false positive
is the most valuable data this system can collect, and it is the only input that
can eventually recalibrate the thresholds. It is persisted to SQLite rather than
held in memory so the demo can be interrupted without losing it.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DOSESENSE_DATA", ROOT / "data" / "synthetic"))
ARTIFACT_DIR = Path(os.environ.get("DOSESENSE_ARTIFACTS", ROOT / "artifacts"))
FRONTEND_DIR = Path(os.environ.get("DOSESENSE_FRONTEND", ROOT / "frontend"))
DB_PATH = Path(os.environ.get("DOSESENSE_DB", ROOT / "artifacts" / "feedback.sqlite3"))

app = FastAPI(
    title="DoseSense",
    version="1.0.0",
    description=(
        "Medication adherence intelligence for care teams. A clinical "
        "decision-support and prioritisation tool, not a diagnostic system. "
        "Every estimate is an inference from indirect signals and every alert "
        "is a request for human review."
    ),
)

STATE = {"engine": None, "metrics": None, "startup_seconds": None, "error": None}


# ---------------------------------------------------------------------------
# Feedback store
# ---------------------------------------------------------------------------

FEEDBACK_KINDS = ("reviewed", "false_positive", "needs_more_evidence", "escalated")


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                snapshot_day INTEGER,
                kind TEXT NOT NULL,
                probability REAL,
                confidence TEXT,
                barrier TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )""")
        # Patient self-reports. Kept in a separate table from clinician feedback
        # because they are a different kind of evidence with a different owner,
        # and because nothing here may ever reach the model - see the note on
        # SelfReportIn below.
        con.execute("""
            CREATE TABLE IF NOT EXISTS self_report (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                report_date TEXT NOT NULL,
                doses TEXT,
                symptom_score REAL,
                barriers TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )""")
        con.execute("CREATE INDEX IF NOT EXISTS ix_self_report_patient"
                    " ON self_report (patient_id, id DESC)")
        con.commit()


# What a patient may tell us about themselves.
#
# Nothing in this table is ever used as a model input. That is not an
# oversight, it is the point: the brief asks for detection *without* asking the
# patient, and a model that learns from self-report inherits exactly the
# unreliability the project exists to route around. What a self-report does is
# let a person confirm, correct or explain a hypothesis the system already
# formed from indirect signals - turning a guess about cost into a fact about
# cost. It is shown to the clinician as separately-sourced evidence and it never
# moves the probability.

SELF_REPORT_DOSES = ("all", "most", "some", "none", "unsure")

SELF_REPORT_BARRIERS = {
    "side_effects": "It made me feel unwell",
    "cost": "It cost more than I expected",
    "supply": "The pharmacy did not have it",
    "travel": "I could not get to the pharmacy",
    "forgot": "I lost track of when to take it",
    "too_many": "There are too many to keep straight",
    "felt_better": "I felt better and stopped",
    "unsure_why": "I am not sure",
}


class SelfReportIn(BaseModel):
    report_date: str | None = Field(default=None, description="ISO date; defaults to today")
    doses: Literal["all", "most", "some", "none", "unsure"] | None = None
    symptom_score: float | None = Field(default=None, ge=0, le=10)
    barriers: list[str] = Field(default_factory=list)
    note: str | None = Field(default=None, max_length=2000)


def record_self_report(patient_id: str, r: SelfReportIn) -> dict:
    bad = [b for b in r.barriers if b not in SELF_REPORT_BARRIERS]
    if bad:
        raise HTTPException(status_code=422, detail=f"unknown barrier tags: {bad}")
    row = {
        "patient_id": patient_id,
        "report_date": r.report_date or time.strftime("%Y-%m-%d"),
        "doses": r.doses,
        "symptom_score": r.symptom_score,
        "barriers": ",".join(r.barriers),
        "note": r.note,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with closing(sqlite3.connect(DB_PATH)) as con:
        cur = con.execute(
            "INSERT INTO self_report (patient_id, report_date, doses, symptom_score,"
            " barriers, note, created_at) VALUES (?,?,?,?,?,?,?)",
            (row["patient_id"], row["report_date"], row["doses"], row["symptom_score"],
             row["barriers"], row["note"], row["created_at"]))
        con.commit()
        row["id"] = cur.lastrowid
    row["barriers"] = list(r.barriers)
    return row


def list_self_reports(patient_id: str, limit: int = 60) -> list[dict]:
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, patient_id, report_date, doses, symptom_score, barriers, note,"
            " created_at FROM self_report WHERE patient_id = ? ORDER BY id DESC LIMIT ?",
            (patient_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["barriers"] = [b for b in (d.get("barriers") or "").split(",") if b]
        d["barrier_labels"] = [SELF_REPORT_BARRIERS[b] for b in d["barriers"]]
        out.append(d)
    return out


class FeedbackIn(BaseModel):
    kind: Literal["reviewed", "false_positive", "needs_more_evidence", "escalated"]
    note: str | None = Field(default=None, max_length=2000)


def record_feedback(patient_id: str, kind: str, note: str | None,
                    snapshot: dict | None) -> dict:
    row = {
        "patient_id": patient_id,
        "snapshot_day": (snapshot or {}).get("day"),
        "kind": kind,
        "probability": ((snapshot or {}).get("inferred") or {}).get("adherence_concern_probability"),
        "confidence": ((snapshot or {}).get("inferred") or {}).get("confidence"),
        "barrier": ((snapshot or {}).get("hypothesis") or {}).get("primary"),
        "note": note,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with closing(sqlite3.connect(DB_PATH)) as con:
        cur = con.execute(
            "INSERT INTO feedback (patient_id, snapshot_day, kind, probability, confidence,"
            " barrier, note, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (row["patient_id"], row["snapshot_day"], row["kind"], row["probability"],
             row["confidence"], row["barrier"], row["note"], row["created_at"]))
        con.commit()
        row["id"] = cur.lastrowid
    return row


def list_feedback(patient_id: str | None = None) -> list[dict]:
    q = "SELECT id, patient_id, snapshot_day, kind, probability, confidence, barrier, note," \
        " created_at FROM feedback"
    args: tuple = ()
    if patient_id:
        q += " WHERE patient_id = ?"
        args = (patient_id,)
    q += " ORDER BY id DESC"
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(q, args).fetchall()]


def feedback_summary() -> dict:
    rows = list_feedback()
    counts: dict[str, int] = {k: 0 for k in FEEDBACK_KINDS}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    reviewed = counts["reviewed"] + counts["false_positive"] + counts["escalated"]
    return {
        "total": len(rows),
        "by_kind": counts,
        # The observed false-positive rate among alerts a human has actually
        # adjudicated. On a live deployment this is the number that should drive
        # threshold changes, and it is the honest replacement for the synthetic
        # ground truth used during development.
        "adjudicated": reviewed,
        "observed_false_positive_rate": (
            round(counts["false_positive"] / reviewed, 4) if reviewed else None),
    }


# ---------------------------------------------------------------------------
# Start-up
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    from dosesense import io, model as M, explain, pipeline

    init_db()
    t0 = time.time()
    try:
        tables = io.load_tables(DATA_DIR, include_ground_truth=False)
        limit = int(os.environ.get("DOSESENSE_PATIENT_LIMIT", "0") or 0)
        if limit and len(tables.get("patients", [])) > limit:
            # Scoring the full panel takes about half a minute. Presenting from a
            # laptop, a smaller cohort starts in a few seconds and shows the same
            # behaviour; the evaluation always runs on the full cohort.
            keep = set(tables["patients"]["patient_id"].head(limit))
            for name, df in tables.items():
                if len(df) and "patient_id" in df.columns:
                    tables[name] = df[df["patient_id"].isin(keep)].reset_index(drop=True)
        if not len(tables.get("patients", [])):
            raise FileNotFoundError(
                f"No cohort found in {DATA_DIR}. Run scripts/generate_data.py first.")
        mdl = M.AdherenceModel.load(ARTIFACT_DIR / "model.pkl")
        engine = pipeline.DoseSenseEngine(tables, mdl, explain.Explainer(mdl))
        STATE["engine"] = engine
        try:
            STATE["metrics"] = io.load_json(ARTIFACT_DIR / "metrics.json")
        except Exception:
            STATE["metrics"] = None
    except Exception as exc:
        STATE["error"] = f"{type(exc).__name__}: {exc}"
    STATE["startup_seconds"] = round(time.time() - t0, 2)


def engine():
    if STATE["engine"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "Engine not ready")
    return STATE["engine"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["system"])
def health() -> dict:
    eng = STATE["engine"]
    return {
        "status": "ok" if eng is not None else "degraded",
        "error": STATE["error"],
        "patients_loaded": len(eng.current) if eng else 0,
        "model_loaded": eng is not None,
        "metrics_loaded": STATE["metrics"] is not None,
        "startup_seconds": STATE["startup_seconds"],
        "disclaimer": ("Synthetic data only. Decision support for prioritisation, "
                       "not a diagnostic device."),
    }


@app.get("/api/overview", tags=["cohort"])
def overview() -> dict:
    eng = engine()
    out = eng.overview()
    out["feedback"] = feedback_summary()
    return out


@app.get("/api/patients", tags=["cohort"])
def patients(
    state: str | None = Query(None, description="Filter by assessment state"),
    tier: str | None = Query(None, description="Filter by priority tier"),
    barrier: str | None = Query(None, description="Filter by inferred barrier"),
    alerts_only: bool = Query(False),
    search: str | None = Query(None, description="Match patient id or condition"),
    limit: int = Query(500, ge=1, le=2000),
) -> dict:
    rows = engine().queue()
    if alerts_only:
        rows = [r for r in rows if r["alert"]]
    if state:
        rows = [r for r in rows if r["state"] == state]
    if tier:
        rows = [r for r in rows if r["priority_tier"] == tier]
    if barrier:
        rows = [r for r in rows if r["barrier"] == barrier]
    if search:
        s = search.lower()
        rows = [r for r in rows
                if s in r["patient_id"].lower() or s in r["condition"].lower()]
    return {"count": len(rows), "patients": rows[:limit]}


@app.get("/api/patients/{patient_id}", tags=["patient"])
def patient(patient_id: str) -> dict:
    d = engine().patient_detail(patient_id)
    if d is None:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id}")
    d["feedback"] = list_feedback(patient_id)
    # Surfaced to the clinician as a fourth, separately-sourced kind of evidence:
    # what the patient said, as distinct from what was recorded about them and
    # what the model inferred.
    d["self_reports"] = list_self_reports(patient_id, limit=20)
    return d


@app.get("/api/patients/{patient_id}/timeline", tags=["patient"])
def timeline(patient_id: str) -> dict:
    eng = engine()
    if patient_id not in eng.timeline:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id}")
    return {"patient_id": patient_id, "timeline": eng.timeline[patient_id]}


@app.get("/api/patients/{patient_id}/risk", tags=["patient"])
def risk(patient_id: str) -> dict:
    eng = engine()
    if patient_id not in eng.current:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id}")
    a = eng.current[patient_id]
    return {
        "patient_id": patient_id,
        "observed": a["observed"],
        "inferred": a["inferred"],
        "hypothesis": a["hypothesis"],
        "action": a["action"],
        "trajectory": eng.trajectory[patient_id],
    }


@app.get("/api/alerts", tags=["cohort"])
def alerts(limit: int = Query(100, ge=1, le=1000)) -> dict:
    rows = [r for r in engine().queue() if r["alert"]][:limit]
    return {"count": len(rows), "alerts": rows}


@app.post("/api/alerts/{patient_id}/feedback", tags=["patient"])
def post_feedback(patient_id: str, body: FeedbackIn) -> dict:
    eng = engine()
    if patient_id not in eng.current:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id}")
    row = record_feedback(patient_id, body.kind, body.note, eng.current[patient_id])
    return {"recorded": row, "summary": feedback_summary()}


@app.get("/api/feedback", tags=["patient"])
def get_feedback() -> dict:
    return {"summary": feedback_summary(), "entries": list_feedback()}


@app.get("/api/metrics", tags=["evaluation"])
def metrics() -> dict:
    if STATE["metrics"] is None:
        raise HTTPException(
            status_code=404,
            detail="No metrics.json found. Run ml/evaluate.py to produce it.")
    return STATE["metrics"]


@app.get("/api/demo", tags=["evaluation"])
def demo() -> dict:
    """A curated walkthrough set, chosen by behaviour rather than hand-picked ids.

    Each case is selected from the live scored cohort by the pattern it
    exhibits, so the walkthrough stays valid if the cohort is regenerated.
    """
    eng = engine()
    rows = eng.queue()

    used: set[str] = set()

    def pick(pred, key=None, reverse=True):
        # Each slot illustrates a different behaviour, so a patient already used
        # is skipped rather than shown twice under two labels.
        cands = [r for r in rows if pred(r) and r["patient_id"] not in used]
        if not cands:
            return None
        if key:
            cands.sort(key=key, reverse=reverse)
        used.add(cands[0]["patient_id"])
        return cands[0]["patient_id"]

    cases = [
        {"slot": "Sustained concern, high consequence",
         "why": "Multiple signal families converge on a patient where a gap matters most.",
         "patient_id": pick(lambda r: r["state"] == "SUSTAINED_CONCERN"
                            and r["clinical_consequence"] >= 0.8
                            and r["evidence_families"] >= 3,
                            key=lambda r: r["priority_score"])},
        {"slot": "Dispensing looks perfect, clinical picture does not",
         "why": ("The case the conventional metric cannot see: proportion of days covered "
                 "is high while the clinical signals have moved."),
         "patient_id": pick(lambda r: r["alert"] and r["pdc_90"] >= 0.85,
                            key=lambda r: r["probability"])},
        {"slot": "Temporary irregularity, deliberately not escalated",
         "why": "One departure from baseline that returned. The system says so and stands down.",
         "patient_id": pick(lambda r: r["state"] == "TEMPORARY_IRREGULARITY",
                            key=lambda r: r["evidence_families"])},
        {"slot": "Access hypothesis",
         "why": "Barrier inference names a reason, with the evidence it rests on.",
         "patient_id": pick(lambda r: r["barrier"] == "ACCESS" and r["alert"],
                            key=lambda r: r["priority_score"])},
        {"slot": "Tolerability hypothesis",
         "why": "Symptoms rose after a regimen change, then collection became less timely.",
         "patient_id": pick(lambda r: r["barrier"] == "SIDE_EFFECT" and r["alert"],
                            key=lambda r: r["priority_score"])},
        {"slot": "Concern with no reason offered",
         "why": ("Evidence supports a concern but nothing points to a cause, so the system "
                 "declines to guess one."),
         "patient_id": pick(lambda r: r["alert"] and r["barrier"] == "UNKNOWN",
                            key=lambda r: r["probability"])},
        {"slot": "Insufficient evidence",
         "why": "A short or sparse record. The system abstains rather than producing a number.",
         "patient_id": pick(lambda r: r["abstained"])},
        {"slot": "Confident negative",
         "why": ("A long complete record with nothing worrying in it. Not the same state as "
                 "'we cannot tell'."),
         "patient_id": pick(lambda r: r["state"] == "WITHIN_BASELINE"
                            and r["evidence_families"] == 0
                            and r["confidence"] == "High",
                            key=lambda r: -r["priority_score"])},
    ]
    return {"cases": [c for c in cases if c["patient_id"]]}


# ---------------------------------------------------------------------------
# Patient portal
# ---------------------------------------------------------------------------


@app.get("/api/patients/{patient_id}/portal", tags=["patient portal"])
def portal(patient_id: str) -> dict:
    """The patient's own view of their care.

    What this deliberately does NOT return: the adherence-concern probability,
    the priority tier, the inferred barrier, or anything derived from them.
    Telling someone there is a 78% chance they are not taking their medicine is
    the accusation this whole project is built to avoid, and a number arrived at
    from indirect signals is not a thing to put in front of the person it is
    about. The clinician sees the estimate; the patient sees their own record,
    their regimen, and an invitation to say how things are going.

    Enforced by test_portal_never_exposes_the_estimate.
    """
    eng = engine()
    if patient_id not in eng.current:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id}")

    p = dict(eng.patients[patient_id])
    for k in ("archetype", "baseline_engagement", "consequence", "enrolled_day",
              "lab_direction", "reports_symptoms", "lab_attendance"):
        p.pop(k, None)

    series = eng._patient_series(patient_id)
    meds = eng.tables["medications"]
    meds = meds[meds["patient_id"] == patient_id].to_dict("records")

    refills = series.get("refills", [])
    last = refills[-1] if refills else None
    supply = int(meds[0]["days_supply"]) if meds else 30
    due = None
    if last:
        from dosesense.datagen import day_to_date
        due = day_to_date(int(last["day"]) + supply).isoformat()

    reports = list_self_reports(patient_id, limit=20)

    # Supportive, non-evaluative framing. No praise for a "good" record either:
    # congratulating someone on high adherence makes the silence that follows a
    # missed week feel like disapproval.
    if reports:
        msg = ("Thanks for keeping us posted. Anything you tell us here goes to your care "
               "team alongside your records.")
    else:
        msg = ("If you have a moment, let us know how you have been getting on. There are no "
               "wrong answers, and it helps your care team understand what would actually help.")

    return {
        "patient": p,
        "medications": [{"name": m["name"], "doses_per_day": m["doses_per_day"],
                         "days_supply": m["days_supply"]} for m in meds],
        "next_refill_due": due,
        "last_collected": last["date"] if last else None,
        "symptom_scale": series.get("symptom_scale"),
        "recent_symptoms": series.get("symptoms", [])[-12:],
        "self_reports": reports,
        "barrier_options": [{"id": k, "label": v} for k, v in SELF_REPORT_BARRIERS.items()],
        "dose_options": list(SELF_REPORT_DOSES),
        "message": msg,
    }


@app.post("/api/patients/{patient_id}/self-report", tags=["patient portal"])
def post_self_report(patient_id: str, body: SelfReportIn) -> dict:
    eng = engine()
    if patient_id not in eng.current:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id}")
    row = record_self_report(patient_id, body)
    return {
        "recorded": row,
        "acknowledgement": "Thanks. Your care team will see this alongside your records.",
        # Stated on every write so the guarantee is visible at the API surface,
        # not only in the documentation.
        "used_for_prediction": False,
    }


@app.get("/api/patients/{patient_id}/self-reports", tags=["patient portal"])
def get_self_reports(patient_id: str) -> dict:
    return {"patient_id": patient_id, "self_reports": list_self_reports(patient_id)}


@app.get("/api/config", tags=["system"])
def get_config() -> dict:
    """The declared policy behind the scores, exposed so it can be inspected."""
    from dosesense import config as C
    return {
        "alert_probability_threshold": C.ALERT_PROBABILITY_THRESHOLD,
        "min_signal_families_for_alert": C.MIN_FAMILIES_FOR_ALERT,
        "snapshot_interval_days": C.SNAPSHOT_INTERVAL_DAYS,
        "label_window_days": C.LABEL_WINDOW_DAYS,
        "priority_tiers": [{"tier": t, "floor": f} for t, f in C.PRIORITY_TIERS],
        "confidence_weights": C.CONFIDENCE_WEIGHT,
        "barrier_labels": C.BARRIER_LABELS,
        "clinical_consequence_by_medication": {
            cond: {m["name"]: m["consequence"] for m in spec["medications"]}
            for cond, spec in C.CONDITIONS.items()
        },
    }


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets") \
        if (FRONTEND_DIR / "assets").exists() else None

    @app.get("/landing", include_in_schema=False)
    def landing():
        f = FRONTEND_DIR / "landing.html"
        if f.exists():
            return FileResponse(str(f))
        return JSONResponse({"detail": "frontend/landing.html not found"}, status_code=404)

    @app.get("/", include_in_schema=False)
    def index():
        f = FRONTEND_DIR / "index.html"
        if f.exists():
            return FileResponse(str(f))
        return JSONResponse({"detail": "frontend/index.html not found"}, status_code=404)
