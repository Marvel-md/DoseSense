"""
Test suite.

The tests are grouped by what could go wrong rather than by module, because the
failure modes of a system like this are specific and a coverage-shaped test
suite would miss most of them.

Four of these tests are the ones that matter:

  test_no_future_leakage        a snapshot must give the same answer whether or
                                not the future exists in the record. This is the
                                only real defence against the silent leakage
                                that makes an adherence model look excellent and
                                perform badly.
  test_pdc_against_hand_worked  PDC computed by hand on a known fill pattern, so
                                the baseline we claim to beat is the real metric
                                and not a weakened version of it.
  test_language_guardrails      no output may assert that a patient did not take
                                a medicine, or label a person. Checked across
                                every assessment the engine produces.
  test_ground_truth_never_served the hidden state and the generator's archetype
                                label must not reach the API surface.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from dosesense import baseline as B            # noqa: E402
from dosesense import barriers as BAR          # noqa: E402
from dosesense import config as C              # noqa: E402
from dosesense import datagen, evaluate as E, explain, model as M, panel as P, pipeline, risk  # noqa: E402
from dosesense.changepoint import detect_change  # noqa: E402
from dosesense.features import (FEATURE_COLUMNS, FEATURE_FAMILIES, build_series,
                                extract_snapshot)  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures. A small cohort, generated once for the whole session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tables():
    return datagen.generate_cohort(n_patients=80, seed=99)


@pytest.fixture(scope="session")
def panel_and_splits(tables):
    pan, detail = P.build_panel(tables, keep_detail=True)
    splits = P.split_patients(pan, seed=99)
    return pan, splits, detail


@pytest.fixture(scope="session")
def trained(panel_and_splits):
    pan, splits, _ = panel_and_splits
    return M.train_model(pan, splits, seed=99)


@pytest.fixture(scope="session")
def engine(tables, trained):
    return pipeline.DoseSenseEngine(tables, trained, explain.Explainer(trained),
                                    history_snapshots=4)


# ---------------------------------------------------------------------------
# Data validity
# ---------------------------------------------------------------------------

class TestData:
    def test_all_tables_populated(self, tables):
        for name in ("patients", "medications", "refills", "appointments",
                     "symptoms", "labs", "events", "ground_truth_states"):
            assert len(tables[name]) > 0, f"{name} is empty"

    def test_every_archetype_present(self, tables):
        got = set(tables["patients"]["archetype"])
        missing = set(datagen.ARCHETYPES) - got
        assert not missing, f"archetypes absent from the cohort: {missing}"

    def test_refill_dates_ordered_and_in_range(self, tables):
        r = tables["refills"]
        assert (r["actual_day"] >= r["expected_day"] - 1).all(), "a fill precedes its due date"
        assert (r["actual_day"] < C.OBSERVATION_DAYS).all(), "a fill falls outside the record"
        assert (r["delay_days"] >= 0).all()

    def test_ground_truth_segments_tile_the_record(self, tables):
        gt = tables["ground_truth_states"]
        for pid, grp in gt.groupby("patient_id"):
            grp = grp.sort_values("start_day")
            assert grp["start_day"].iloc[0] == 0, f"{pid} trajectory does not start at day 0"
            assert grp["end_day"].iloc[-1] == C.OBSERVATION_DAYS - 1
            ends = grp["end_day"].to_numpy()[:-1]
            starts = grp["start_day"].to_numpy()[1:]
            assert np.all(starts == ends + 1), f"{pid} has a gap or overlap in its trajectory"

    def test_adherent_archetypes_are_negative_by_construction(self, panel_and_splits):
        pan, _, _ = panel_and_splits
        for archetype in ("ADHERENT_STABLE", "MISLEADING_ANOMALY", "DISEASE_PROGRESSION"):
            sel = pan[pan["archetype"] == archetype]
            if not len(sel):
                continue
            assert sel["label"].max() == 0, (
                f"{archetype} produced a positive label; it is meant to be a clean negative "
                f"so that the false-alert rate on it means something")

    def test_silent_nonadherence_has_healthy_dispensing(self, panel_and_splits):
        """The whole point of this archetype is that the pharmacy record looks fine."""
        pan, _, _ = panel_and_splits
        sel = pan[(pan["archetype"] == "SILENT_NONADHERENCE") & (pan["label"] == 1)]
        if len(sel) < 5:
            pytest.skip("too few silent-non-adherence snapshots in this small cohort")
        assert sel["refill_pdc_90"].mean() > 0.75, (
            "silent non-adherence should present with a high proportion of days covered")
        assert sel["baseline_flag"].mean() < 0.30, (
            "the PDC rule should largely miss this archetype, which is why it exists")


# ---------------------------------------------------------------------------
# Causality: the test that protects every metric we report
# ---------------------------------------------------------------------------

class TestCausality:
    def test_no_future_leakage(self, tables):
        """A snapshot must be unchanged by truncating everything after it.

        Recomputed against a record trimmed to day t, every feature must match
        the value computed against the full record. If any window is accidentally
        centred or forward-looking this fails, and it is the only check that
        would catch it.
        """
        series = build_series(tables)
        pid = sorted(series)[3]
        ps = series[pid]
        t = 360

        full = extract_snapshot(ps, t)

        trimmed_tables = {}
        for name, df in tables.items():
            if not len(df):
                trimmed_tables[name] = df
                continue
            col = ("actual_day" if name == "refills"
                   else "scheduled_day" if name == "appointments"
                   else "day" if "day" in df.columns else None)
            trimmed_tables[name] = df[df[col] <= t] if col else df
        trimmed = extract_snapshot(build_series(trimmed_tables)[pid], t)

        for key in FEATURE_COLUMNS:
            a, b = full["features"][key], trimmed["features"][key]
            assert a == pytest.approx(b, abs=1e-9), (
                f"feature {key} changed when the future was removed "
                f"({a} vs {b}) — this is data leakage")

    def test_snapshot_only_sees_prior_records(self, tables):
        series = build_series(tables)
        pid = sorted(series)[0]
        ps = series[pid]
        early = extract_snapshot(ps, 210)
        assert early["features"]["refill_n_fills_365"] <= np.sum(ps.fill_day <= 210)

    def test_splits_share_no_patients(self, panel_and_splits):
        pan, splits, _ = panel_and_splits
        ids = {k: set(pan.loc[splits[k], "patient_id"]) for k in ("train", "calib", "test")}
        assert not ids["train"] & ids["test"]
        assert not ids["train"] & ids["calib"]
        assert not ids["calib"] & ids["test"]


# ---------------------------------------------------------------------------
# Baseline arithmetic
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_pdc_against_hand_worked(self):
        """Three 30-day fills at days 0, 30 and 90, over a 120-day window.

        Coverage is days 0-59 and 90-119: ninety covered days out of 120, so PDC
        is 0.75. Days 60-89 are uncovered because the third fill arrived a month
        late. Worked by hand so the comparator is the genuine metric.
        """
        fills, supply = [0, 30, 90], [30, 30, 30]
        assert B.pdc(fills, supply, 0, 120) == pytest.approx(0.75)
        assert B.coverage_days(fills, supply, 0, 120) == 90

    def test_pdc_caps_at_one_and_mpr_does_not(self):
        """Stockpiling: four 30-day fills two weeks apart inside a 60-day window."""
        fills, supply = [0, 14, 28, 42], [30, 30, 30, 30]
        assert B.pdc(fills, supply, 0, 60) == pytest.approx(1.0)
        assert B.mpr(fills, supply, 0, 60) == pytest.approx(2.0)

    def test_pdc_handles_overlapping_supply_as_union(self):
        fills, supply = [0, 10], [30, 30]
        assert B.coverage_days(fills, supply, 0, 60) == 40

    def test_empty_and_degenerate_windows(self):
        assert B.coverage_days([], [], 0, 30) == 0
        assert np.isnan(B.pdc([0], [30], 30, 30))

    def test_baseline_flags_below_threshold(self):
        low = B.baseline_score(0.55, 2.0)
        high = B.baseline_score(0.97, 1.0)
        assert low["flag"] == 1 and high["flag"] == 0
        assert low["score"] > high["score"]

    def test_baseline_gap_rule_fires_independently_of_pdc(self):
        r = B.baseline_score(0.95, 22.0)
        assert r["flag"] == 1, "a 22-day gap should trip the gap rule even at a healthy PDC"


# ---------------------------------------------------------------------------
# Change detection: the temporary-versus-meaningful requirement
# ---------------------------------------------------------------------------

class TestChangeDetection:
    def test_stable_series_reports_stable(self):
        rng = np.random.default_rng(0)
        r = detect_change(rng.normal(2.0, 0.4, 16), min_absolute_effect=3.5)
        assert r.status == C.CHANGE_STABLE

    def test_sustained_shift_reports_persistent(self):
        values = [1, 2, 1, 2, 1, 1, 2, 18, 20, 17, 19, 21, 18]
        r = detect_change(values, min_absolute_effect=3.5)
        assert r.status == C.CHANGE_PERSISTENT
        assert r.change_index == 7
        assert not r.reverted

    def test_single_outlier_that_reverts_is_not_persistent(self):
        """The holiday case. One large gap, then back to normal."""
        values = [1, 2, 1, 2, 1, 2, 1, 24, 2, 1, 2, 1, 2]
        r = detect_change(values, min_absolute_effect=3.5)
        assert r.status != C.CHANGE_PERSISTENT, (
            "an isolated gap that fully reverted must not be reported as a sustained shift")

    def test_shift_detected_too_recently_is_provisional(self):
        values = [1, 2, 1, 2, 1, 2, 1, 2, 19, 21]
        r = detect_change(values, min_absolute_effect=3.5)
        assert r.provisional, "with two points after the change, reversion cannot yet be judged"

    def test_clinical_floor_suppresses_trivial_shifts(self):
        """A shift from half a day late to two days late is real and irrelevant."""
        values = [0.4, 0.5, 0.4, 0.6, 0.5, 2.0, 2.1, 1.9, 2.2, 2.0]
        assert detect_change(values, min_absolute_effect=0.0).status == C.CHANGE_PERSISTENT
        assert detect_change(values, min_absolute_effect=3.5).status == C.CHANGE_STABLE

    def test_direction_respected_for_inverted_markers(self):
        """For INR or peak flow, falling is the adverse direction."""
        falling = [2.6, 2.5, 2.7, 2.6, 2.5, 1.0, 0.9, 1.1, 1.0, 0.9]
        assert detect_change(falling, higher_is_worse=False).effect_sd > 0
        assert detect_change(falling, higher_is_worse=True).status == C.CHANGE_STABLE

    def test_short_series_returns_stable_not_an_error(self):
        assert detect_change([1, 2, 3]).status == C.CHANGE_STABLE
        assert detect_change([]).status == C.CHANGE_STABLE


# ---------------------------------------------------------------------------
# Uncertainty and abstention
# ---------------------------------------------------------------------------

class TestUncertainty:
    def _strength(self, n_ev, n_avail=None):
        return {"n_families_with_evidence": n_ev,
                "n_families_available": n_avail if n_avail is not None else max(n_ev, 4),
                "families_available": [], "families_with_evidence": [],
                "total_evidence_weight": float(n_ev)}

    def test_sparse_record_abstains(self):
        r = risk.assess_confidence(0.8, 0.02, self._strength(3),
                                   {"refill_n_fills_365": 1, "appt_n_365": 0, "lab_n_results": 0})
        assert r["abstained"] and r["confidence"] == C.CONFIDENCE_INSUFFICIENT

    def test_high_probability_on_one_signal_abstains(self):
        r = risk.assess_confidence(0.85, 0.02, self._strength(1),
                                   {"refill_n_fills_365": 12, "appt_n_365": 4, "lab_n_results": 6})
        assert r["abstained"], "an elevated estimate on a single signal family must not alert"

    def test_clean_record_with_no_findings_is_a_confident_negative(self):
        """The distinction that matters: 'nothing wrong' is not 'cannot tell'."""
        r = risk.assess_confidence(0.03, 0.01, self._strength(0, n_avail=6),
                                   {"refill_n_fills_365": 13, "appt_n_365": 5, "lab_n_results": 6})
        assert not r["abstained"]
        assert r["confidence"] == C.CONFIDENCE_HIGH

    def test_ensemble_disagreement_lowers_confidence(self):
        feats = {"refill_n_fills_365": 12, "appt_n_365": 4, "lab_n_results": 6}
        calm = risk.assess_confidence(0.8, 0.005, self._strength(4), feats)
        noisy = risk.assess_confidence(0.8, 0.20, self._strength(4), feats)
        assert calm["confidence_score"] > noisy["confidence_score"]

    def test_priority_weighs_consequence_not_probability_alone(self):
        """A lower-probability concern on a critical medicine must outrank a
        higher-probability concern on a benign one."""
        warfarin = risk.priority(0.61, 0.95, C.CONFIDENCE_HIGH)
        thyroid = risk.priority(0.88, 0.40, C.CONFIDENCE_HIGH)
        assert warfarin["priority_score"] > thyroid["priority_score"]

    def test_abstention_suppresses_priority(self):
        confident = risk.priority(0.9, 0.9, C.CONFIDENCE_HIGH)
        abstained = risk.priority(0.9, 0.9, C.CONFIDENCE_INSUFFICIENT)
        assert abstained["priority_score"] < confident["priority_score"]

    def test_alert_states_are_distinct(self):
        assert not risk.should_alert(0.9, C.CONFIDENCE_INSUFFICIENT)["alert"]
        assert risk.should_alert(0.8, C.CONFIDENCE_HIGH, C.CHANGE_PERSISTENT)["state"] == "SUSTAINED_CONCERN"
        assert risk.should_alert(0.2, C.CONFIDENCE_HIGH, C.CHANGE_TEMPORARY)["state"] == "TEMPORARY_IRREGULARITY"
        assert risk.should_alert(0.8, C.CONFIDENCE_HIGH, C.CHANGE_TEMPORARY)["state"] == "RESOLVING"


# ---------------------------------------------------------------------------
# Barrier inference
# ---------------------------------------------------------------------------

class TestBarriers:
    BASE = {k: 0.0 for k in FEATURE_COLUMNS}
    PATIENT = {"patient_id": "T1", "insurance_tier": "Public",
               "distance_to_pharmacy_km": 2.0, "consequence": 0.6}

    def test_abstains_without_evidence(self):
        r = BAR.infer_barriers(dict(self.BASE), self.PATIENT, [], 400)
        assert r["primary"] == "UNKNOWN"
        assert r["evidence"] == []
        assert "not point clearly" in r["note"] or "Ask the patient" in r["note"]

    def test_supply_events_support_access(self):
        f = dict(self.BASE, refill_overdue_ratio=1.8, refill_delay_excess=9.0,
                 refill_consecutive_late=3)
        events = [{"day": 330, "event_type": "pharmacy_supply_issue", "detail": "out of stock"}]
        r = BAR.infer_barriers(f, self.PATIENT, events, 400)
        assert r["primary"] == "ACCESS"
        assert r["evidence"], "a named barrier must carry the evidence it rests on"

    def test_insurance_change_supports_cost(self):
        f = dict(self.BASE, refill_delay_excess=7.0, rx_n_medications=2,
                 refill_consecutive_late=2)
        events = [{"day": 320, "event_type": "insurance_change", "detail": "plan changed"},
                  {"day": 340, "event_type": "prescription_abandoned", "detail": "not collected"}]
        r = BAR.infer_barriers(f, dict(self.PATIENT, insurance_tier="Self-pay"), events, 400)
        assert r["primary"] == "COST"

    def test_symptom_rise_after_regimen_change_supports_tolerability(self):
        f = dict(self.BASE, sym_delta=2.4, refill_delay_excess=4.0)
        events = [{"day": 340, "event_type": "medication_added", "detail": "second agent"}]
        r = BAR.infer_barriers(f, self.PATIENT, events, 400,
                               {"symptom": {"status": C.CHANGE_PERSISTENT}})
        assert r["primary"] == "SIDE_EFFECT"

    def test_missed_followups_support_care_engagement(self):
        f = dict(self.BASE, appt_consecutive_missed=3, appt_days_since_attended=260,
                 appt_missed_rate_180=0.8, appt_missed_rate_excess=0.4)
        r = BAR.infer_barriers(f, self.PATIENT, [], 400)
        assert r["primary"] == "CARE_ENGAGEMENT"

    def test_all_scores_exposed_for_inspection(self):
        r = BAR.infer_barriers(dict(self.BASE), self.PATIENT, [], 400)
        assert set(r["all_scores"]) == set(C.BARRIERS) - {"UNKNOWN"}


# ---------------------------------------------------------------------------
# Model and metrics
# ---------------------------------------------------------------------------

class TestModel:
    def test_predictions_are_probabilities(self, panel_and_splits, trained):
        pan, splits, _ = panel_and_splits
        X = pan[splits["test"]][trained.feature_columns].to_numpy(dtype=float)
        out = trained.predict(X)
        assert out["probability"].min() >= 0 and out["probability"].max() <= 1
        assert (out["ensemble_sd"] >= 0).all()

    def test_beats_the_baseline_on_held_out_patients(self, panel_and_splits, trained):
        pan, splits, _ = panel_and_splits
        te = pan[splits["test"]]
        y = te["label"].to_numpy()
        prob = trained.predict(te[trained.feature_columns].to_numpy(dtype=float))["probability"]
        model_m = E.classification_metrics(y, prob, threshold=0.5)
        base_m = E.classification_metrics(y, te["baseline_score"].to_numpy(),
                                          flag=te["baseline_flag"].to_numpy())
        assert model_m["pr_auc"] > base_m["pr_auc"], (
            "the model must beat the conventional PDC comparator or it has no reason to exist")

    def test_calibration_is_reasonable(self, panel_and_splits, trained):
        pan, splits, _ = panel_and_splits
        te = pan[splits["test"]]
        prob = trained.predict(te[trained.feature_columns].to_numpy(dtype=float))["probability"]
        cal = E.calibration_curve(te["label"].to_numpy(), prob)
        assert cal["expected_calibration_error"] < 0.12

    def test_save_and_load_round_trip(self, trained, tmp_path):
        path = tmp_path / "m.pkl"
        trained.save(path)
        loaded = M.AdherenceModel.load(path)
        X = np.zeros((3, len(trained.feature_columns)))
        assert np.allclose(trained.predict_proba(X), loaded.predict_proba(X))

    def test_every_feature_belongs_to_exactly_one_family(self):
        seen: dict[str, str] = {}
        for fam, feats in FEATURE_FAMILIES.items():
            for f in feats:
                assert f not in seen, f"{f} appears in both {seen.get(f)} and {fam}"
                seen[f] = fam
        assert set(seen) == set(FEATURE_COLUMNS)

    def test_protected_attributes_are_not_model_inputs(self):
        """Sex and insurance tier are held back for evaluation, never predicted from."""
        for banned in ("sex", "insurance_tier", "distance_to_pharmacy", "consequence", "archetype"):
            assert not any(banned in c for c in FEATURE_COLUMNS), (
                f"{banned} leaked into the model feature matrix")

    def test_explainer_produces_named_contributions(self, trained):
        ex = explain.Explainer(trained)
        out = ex.explain_row(np.zeros(len(trained.feature_columns)))
        assert out["contributions"] and out["by_family"]
        for c in out["contributions"]:
            assert c["label"] != c["feature"] or c["feature"] in explain.FEATURE_LABELS, (
                f"{c['feature']} has no plain-language label")

    def test_all_features_have_plain_language_labels(self):
        missing = [c for c in FEATURE_COLUMNS if c not in explain.FEATURE_LABELS]
        assert not missing, f"features shown to clinicians without a readable name: {missing}"


class TestEvaluationMetrics:
    def test_lead_time_prefers_earlier_detection(self, panel_and_splits):
        pan, splits, _ = panel_and_splits
        te = pan[splits["test"]].reset_index(drop=True)
        perfect = te["label"].to_numpy()
        r = E.detection_lead_time(te, perfect)
        assert r["median_days_to_detection"] <= 0

    def test_alert_burden_counts_only_false_positives(self, panel_and_splits):
        pan, splits, _ = panel_and_splits
        te = pan[splits["test"]].reset_index(drop=True)
        r = E.alert_burden(te, te["label"].to_numpy())
        assert r["false_alerts"] == 0
        assert r["precision_of_alerts"] == 1.0

    def test_subgroups_flag_small_samples_instead_of_guessing(self, panel_and_splits):
        pan, splits, _ = panel_and_splits
        te = pan[splits["test"]].reset_index(drop=True)
        prob = np.full(len(te), 0.5)
        rows = E.subgroup_metrics(te, prob, "condition", min_n=10_000)
        assert all("note" in r for r in rows)


# ---------------------------------------------------------------------------
# Safety: language and information hygiene
# ---------------------------------------------------------------------------

def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)


class TestSafety:
    def test_language_guardrails(self, engine):
        """No output may assert a missed dose or label a person."""
        offences = []
        for pid, a in engine.current.items():
            for s in _walk_strings(a):
                low = s.lower()
                for phrase in C.FORBIDDEN_PHRASES:
                    if phrase in low:
                        offences.append((pid, phrase, s))
        assert not offences, f"prohibited phrasing in output: {offences[:3]}"

    def test_probabilities_are_framed_as_possible_concerns(self, engine):
        a = next(iter(engine.current.values()))
        assert "adherence_concern_probability" in a["inferred"]
        assert "hypothesis" in a, "a barrier must be served as a hypothesis, not a finding"

    def test_ground_truth_never_served(self, engine):
        """The generator's archetype and hidden state must not reach the API."""
        for pid in list(engine.current)[:25]:
            d = engine.patient_detail(pid)
            assert "archetype" not in d["patient"]
            blob = " ".join(_walk_strings(d))
            for state in C.STATES:
                assert state not in blob, f"hidden latent state {state} leaked for {pid}"

    def test_observed_inferred_hypothesis_are_separate_keys(self, engine):
        a = next(iter(engine.current.values()))
        assert set(("observed", "inferred", "hypothesis", "action")) <= set(a)
        assert "adherence_concern_probability" not in a["observed"], (
            "a model estimate must never sit inside the observed-facts block")

    def test_facts_are_only_things_actually_measured(self, engine):
        for a in list(engine.current.values())[:40]:
            for fact in a["observed"]["facts"]:
                assert fact["family"] in C.SIGNAL_FAMILIES
                assert 0 <= fact["weight"] <= 1


# ---------------------------------------------------------------------------
# Engine and API
# ---------------------------------------------------------------------------

class TestEngine:
    def test_queue_is_ordered_by_priority(self, engine):
        q = engine.queue()
        scores = [r["priority_score"] for r in q]
        assert scores == sorted(scores, reverse=True)

    def test_every_patient_is_scored(self, engine, tables):
        assert len(engine.current) == len(tables["patients"])

    def test_trajectory_is_chronological(self, engine):
        for traj in list(engine.trajectory.values())[:20]:
            days = [s["day"] for s in traj]
            assert days == sorted(days)

    def test_timeline_is_chronological_and_marks_inferences(self, engine):
        pid = engine.queue()[0]["patient_id"]
        tl = engine.timeline[pid]
        assert [e["day"] for e in tl] == sorted(e["day"] for e in tl)
        for e in tl:
            if e["kind"] == "inference":
                assert e["observed"] is False

    def test_overview_counts_are_consistent(self, engine):
        o = engine.overview()
        assert o["requiring_review"] == sum(1 for r in engine.queue() if r["alert"])
        assert o["critical"] + o["high"] <= o["requiring_review"]

    def test_unknown_patient_returns_none(self, engine):
        assert engine.patient_detail("NOT_A_PATIENT") is None


class TestAPI:
    @pytest.fixture(scope="class")
    def client(self, tmp_path_factory, tables, trained):
        """Serve a small cohort from a temporary directory."""
        from fastapi.testclient import TestClient
        from dosesense import io

        d = tmp_path_factory.mktemp("api")
        io.save_tables(tables, d / "data")
        trained.save(d / "artifacts" / "model.pkl")

        import os
        os.environ["DOSESENSE_DATA"] = str(d / "data")
        os.environ["DOSESENSE_ARTIFACTS"] = str(d / "artifacts")
        os.environ["DOSESENSE_DB"] = str(d / "feedback.sqlite3")
        for mod in [m for m in list(sys.modules) if m.startswith("backend")]:
            del sys.modules[mod]
        from backend.app.main import app
        with TestClient(app) as c:
            yield c

    def test_health(self, client):
        r = client.get("/api/health").json()
        assert r["status"] == "ok"
        assert r["patients_loaded"] > 0
        assert "not a diagnostic device" in r["disclaimer"]

    def test_overview_and_queue(self, client):
        assert client.get("/api/overview").status_code == 200
        q = client.get("/api/patients?limit=10").json()
        assert q["count"] > 0 and len(q["patients"]) <= 10

    def test_filters_narrow_the_queue(self, client):
        allq = client.get("/api/patients?limit=2000").json()["count"]
        alerts = client.get("/api/patients?alerts_only=true&limit=2000").json()["count"]
        assert alerts <= allq

    def test_patient_detail_and_subresources(self, client):
        pid = client.get("/api/patients?limit=1").json()["patients"][0]["patient_id"]
        d = client.get(f"/api/patients/{pid}").json()
        assert set(("patient", "medications", "assessment", "trajectory", "timeline", "series")) <= set(d)
        assert client.get(f"/api/patients/{pid}/timeline").status_code == 200
        assert client.get(f"/api/patients/{pid}/risk").status_code == 200

    def test_missing_patient_is_404(self, client):
        assert client.get("/api/patients/NOPE").status_code == 404
        assert client.get("/api/patients/NOPE/risk").status_code == 404

    def test_feedback_round_trip(self, client):
        pid = client.get("/api/patients?limit=1").json()["patients"][0]["patient_id"]
        r = client.post(f"/api/alerts/{pid}/feedback",
                        json={"kind": "false_positive", "note": "patient was travelling"})
        assert r.status_code == 200
        body = r.json()
        assert body["recorded"]["kind"] == "false_positive"
        assert body["summary"]["total"] >= 1
        hist = client.get(f"/api/patients/{pid}").json()["feedback"]
        assert any(f["kind"] == "false_positive" for f in hist)

    def test_feedback_rejects_unknown_kind(self, client):
        pid = client.get("/api/patients?limit=1").json()["patients"][0]["patient_id"]
        r = client.post(f"/api/alerts/{pid}/feedback", json={"kind": "looks_bad"})
        assert r.status_code == 422

    def test_demo_cases_resolve_to_real_patients(self, client):
        cases = client.get("/api/demo").json()["cases"]
        assert len(cases) >= 4
        ids = [c["patient_id"] for c in cases]
        assert len(ids) == len(set(ids)), "a demo slot is reusing a patient"
        for pid in ids:
            assert client.get(f"/api/patients/{pid}").status_code == 200

    def test_config_exposes_declared_policy(self, client):
        c = client.get("/api/config").json()
        assert "clinical_consequence_by_medication" in c
        assert c["alert_probability_threshold"] == C.ALERT_PROBABILITY_THRESHOLD

    def test_openapi_schema_is_valid(self, client):
        s = client.get("/openapi.json").json()
        assert "/api/patients/{patient_id}" in s["paths"]
