"""
Clinical policy and simulation configuration for DoseSense.

Everything in this module is *declared policy*, not learned behaviour. Keeping it
separate from the ML code matters for two reasons:

1. A care team must be able to audit and change the consequence weighting without
   retraining a model. Clinical severity is a governance decision, not a
   statistical one.
2. It makes the boundary between "what the model inferred" and "what we told the
   system to care about" inspectable, which is a requirement of the brief.

All values are illustrative defaults for a synthetic demonstration. They are not
validated clinical thresholds and must not be used for care decisions.
"""

from __future__ import annotations

RANDOM_SEED = 20260910

# ---------------------------------------------------------------------------
# Latent adherence states (hidden during inference, used only for evaluation)
# ---------------------------------------------------------------------------

STATE_ADHERENT = "ADHERENT"
STATE_PARTIAL = "PARTIALLY_ADHERENT"
STATE_LAPSED = "LAPSED"
STATES = (STATE_ADHERENT, STATE_PARTIAL, STATE_LAPSED)

# Fraction of prescribed doses actually taken, by latent state.
STATE_TAKING_FRACTION = {
    STATE_ADHERENT: (0.92, 1.00),
    STATE_PARTIAL: (0.55, 0.80),
    STATE_LAPSED: (0.00, 0.35),
}

# ---------------------------------------------------------------------------
# Conditions, their medications, and the lab marker that tracks control
# ---------------------------------------------------------------------------
# consequence: clinical seriousness of a sustained adherence gap for this
#   condition/drug pairing, on 0-1. Encodes "a missed anticoagulant is not a
#   missed multivitamin".
# lab_direction: +1 means the marker rises as control worsens.

CONDITIONS = {
    "Type 2 diabetes": {
        "medications": [
            {"name": "Metformin 1000mg", "doses_per_day": 2, "days_supply": 30, "consequence": 0.55},
            {"name": "Empagliflozin 10mg", "doses_per_day": 1, "days_supply": 30, "consequence": 0.50},
        ],
        "lab": {"name": "HbA1c", "unit": "%", "baseline": (6.6, 8.4), "worsen_per_month": 0.22,
                "noise": 0.16, "direction": 1, "interval_days": 90, "floor": 5.2, "ceiling": 14.0},
        "symptom_scale": "Diabetes symptom burden (0-10)",
        "symptom_lag_days": 30,
    },
    "Hypertension": {
        "medications": [
            {"name": "Amlodipine 5mg", "doses_per_day": 1, "days_supply": 30, "consequence": 0.60},
            {"name": "Telmisartan 40mg", "doses_per_day": 1, "days_supply": 30, "consequence": 0.60},
        ],
        "lab": {"name": "Systolic BP", "unit": "mmHg", "baseline": (126, 146), "worsen_per_month": 3.1,
                "noise": 5.5, "direction": 1, "interval_days": 45, "floor": 100, "ceiling": 205},
        "symptom_scale": "Headache / dizziness burden (0-10)",
        "symptom_lag_days": 18,
    },
    "Heart failure": {
        "medications": [
            {"name": "Furosemide 40mg", "doses_per_day": 1, "days_supply": 30, "consequence": 0.85},
            {"name": "Carvedilol 6.25mg", "doses_per_day": 2, "days_supply": 30, "consequence": 0.80},
        ],
        "lab": {"name": "NT-proBNP", "unit": "pg/mL", "baseline": (320, 900), "worsen_per_month": 145.0,
                "noise": 90.0, "direction": 1, "interval_days": 60, "floor": 90, "ceiling": 6000},
        "symptom_scale": "Breathlessness / oedema burden (0-10)",
        "symptom_lag_days": 12,
    },
    "Atrial fibrillation": {
        "medications": [
            {"name": "Warfarin 5mg", "doses_per_day": 1, "days_supply": 28, "consequence": 0.95},
            {"name": "Bisoprolol 2.5mg", "doses_per_day": 1, "days_supply": 30, "consequence": 0.65},
        ],
        "lab": {"name": "INR", "unit": "ratio", "baseline": (2.1, 2.8), "worsen_per_month": -0.30,
                "noise": 0.22, "direction": -1, "interval_days": 30, "floor": 0.9, "ceiling": 5.0},
        "symptom_scale": "Palpitation / fatigue burden (0-10)",
        "symptom_lag_days": 15,
    },
    "Hypothyroidism": {
        "medications": [
            {"name": "Levothyroxine 75mcg", "doses_per_day": 1, "days_supply": 30, "consequence": 0.40},
        ],
        "lab": {"name": "TSH", "unit": "mIU/L", "baseline": (1.4, 3.6), "worsen_per_month": 1.35,
                "noise": 0.55, "direction": 1, "interval_days": 90, "floor": 0.3, "ceiling": 22.0},
        "symptom_scale": "Fatigue / cold intolerance burden (0-10)",
        "symptom_lag_days": 40,
    },
    "Asthma": {
        "medications": [
            {"name": "Budesonide/Formoterol inhaler", "doses_per_day": 2, "days_supply": 30, "consequence": 0.70},
        ],
        "lab": {"name": "Peak expiratory flow", "unit": "L/min", "baseline": (370, 460),
                "worsen_per_month": -16.0, "noise": 22.0, "direction": -1, "interval_days": 60,
                "floor": 180, "ceiling": 560},
        "symptom_scale": "Wheeze / rescue-inhaler burden (0-10)",
        "symptom_lag_days": 10,
    },
}

# ---------------------------------------------------------------------------
# Barrier taxonomy
# ---------------------------------------------------------------------------

BARRIERS = (
    "ACCESS",
    "COST",
    "SIDE_EFFECT",
    "REGIMEN_COMPLEXITY",
    "CARE_ENGAGEMENT",
    "UNKNOWN",
)

BARRIER_LABELS = {
    "ACCESS": "Access or refill disruption",
    "COST": "Cost or affordability pressure",
    "SIDE_EFFECT": "Tolerability or side-effect concern",
    "REGIMEN_COMPLEXITY": "Regimen complexity",
    "CARE_ENGAGEMENT": "Disengagement from follow-up care",
    "UNKNOWN": "Insufficient evidence to infer a barrier",
}

# Minimum barrier score before we are willing to name a barrier at all.
BARRIER_MIN_SCORE = 0.34
# Minimum margin over the runner-up before we present a single barrier
# rather than two competing hypotheses.
BARRIER_MIN_MARGIN = 0.08

# ---------------------------------------------------------------------------
# Evaluation snapshot geometry
# ---------------------------------------------------------------------------

OBSERVATION_DAYS = 540          # length of each patient's simulated record
SNAPSHOT_INTERVAL_DAYS = 30     # how often the system re-scores a patient
SNAPSHOT_WARMUP_DAYS = 180      # need history before a personal baseline exists
LABEL_WINDOW_DAYS = 90          # trailing window the ground-truth label describes
# A snapshot is labelled a concern when the patient spent at least this share of
# the trailing window in a non-adherent latent state. Set high enough that a
# single two-week slip does NOT become a positive label.
LABEL_MIN_NONADHERENT_FRACTION = 0.34

# ---------------------------------------------------------------------------
# Uncertainty and abstention
# ---------------------------------------------------------------------------

# Independent signal families. Evidence strength counts how many are informative.
SIGNAL_FAMILIES = ("refill", "prescription", "appointment", "symptom", "laboratory", "behavioural")

MIN_FAMILIES_FOR_ALERT = 2      # below this we abstain rather than alert
ALERT_PROBABILITY_THRESHOLD = 0.50

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
CONFIDENCE_INSUFFICIENT = "Insufficient evidence"

# ---------------------------------------------------------------------------
# Priority tiers: likelihood x consequence x confidence
# ---------------------------------------------------------------------------

PRIORITY_TIERS = (
    ("CRITICAL", 0.52),
    ("HIGH", 0.34),
    ("MODERATE", 0.18),
    ("LOW", 0.0),
)

CONFIDENCE_WEIGHT = {
    CONFIDENCE_HIGH: 1.00,
    CONFIDENCE_MEDIUM: 0.78,
    CONFIDENCE_LOW: 0.50,
    CONFIDENCE_INSUFFICIENT: 0.22,
}

# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------

CP_MIN_SEGMENT = 3              # minimum points either side of a change point
CP_PENALTY_SCALE = 2.4          # BIC-style penalty multiplier
CP_REVERSION_TOLERANCE = 0.55   # shift counts as reverted if it decays below this
CP_MIN_EFFECT_SD = 1.1          # shift must exceed this many baseline SDs

CHANGE_STABLE = "STABLE"
CHANGE_TEMPORARY = "TEMPORARY_ANOMALY"
CHANGE_PERSISTENT = "PERSISTENT_SHIFT"

# ---------------------------------------------------------------------------
# Language guardrails
# ---------------------------------------------------------------------------
# Enforced by tests/test_language.py. The system must never assert that a
# patient did not take a medication, and must never label a person.

FORBIDDEN_PHRASES = (
    "non-compliant",
    "noncompliant",
    "definitely missed",
    "did not take",
    "failed to take",
    "patient is refusing",
    "confirmed non-adherence",
)
