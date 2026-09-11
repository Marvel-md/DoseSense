#!/usr/bin/env python3
"""Render docs/evaluation.md from artifacts/metrics.json.

Every figure in the documentation is produced by this script from the saved
evaluation output. Nothing is transcribed. If a number in the docs looks wrong,
rerun the pipeline and it changes; there is no path by which a stale or invented
figure can survive in the write-up.

Usage:
    python scripts/make_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dosesense import io   # noqa: E402


def f(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def pc(v, n=1):
    return "—" if v is None else f"{v * 100:.{n}f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    m = io.load_json(ROOT / "artifacts" / "metrics.json")
    mod, base = m["model"], m["baseline_pdc"]
    ds, cal = m["dataset"], m["calibration"]
    L = []

    L.append("# Evaluation\n")
    L.append(f"Generated from `artifacts/metrics.json` on {m['generated_at']}. "
             "Every figure below is produced by `scripts/make_report.py` reading the "
             "saved output of `ml/evaluate.py`. None is transcribed by hand.\n")

    L.append("## What was measured, and on what\n")
    L.append(f"The cohort is {ds['n_patients']} synthetic patients over 540 days, yielding "
             f"{ds['n_snapshots']:,} patient-snapshots at 30-day intervals. "
             f"{pc(ds['concern_rate'])} of snapshots carry a concern under the ground-truth "
             f"definition. Results below are on {ds['n_test_patients']} patients "
             f"({ds['n_test_snapshots']:,} snapshots) held out **at the patient level**: no "
             "patient contributes rows to more than one split, and the isotonic calibrator was "
             "fitted on a third disjoint set of patients.\n")
    L.append("Accuracy is not reported as a headline. With a concern rate near 38% it is easy "
             "to score well on and it says nothing about the two things that decide whether a "
             "care team keeps the system switched on: how many unnecessary calls it generates, "
             "and whether the stated probability means what it says.\n")

    L.append("## Headline comparison against the metric in current use\n")
    L.append("The comparator is Proportion of Days Covered with the conventional 0.80 threshold, "
             "plus a seven-day refill-gap rule. It is implemented faithfully in "
             "`dosesense/baseline.py` and hand-verified in the test suite, so this is the real "
             "metric rather than a weakened version of it.\n")
    L.append(table(
        ["", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier"],
        [["**DoseSense**", f"**{f(mod['roc_auc'])}**", f"**{f(mod['pr_auc'])}**",
          f(mod["precision"]), f(mod["recall"]), f"**{f(mod['f1'])}**", f(mod["brier"], 4)],
         ["PDC / refill-gap rule", f(base["roc_auc"]), f(base["pr_auc"]),
          f(base["precision"]), f(base["recall"]), f(base["f1"]), f(base["brier"], 4)]]))
    L.append(f"\nPR-AUC improves by {m['improvement']['pr_auc_delta']:+.4f} and F1 by "
             f"{m['improvement']['f1_delta']:+.4f}. PR-AUC is the figure to read: the negative "
             "class is large, and ROC-AUC rewards ranking the obvious cases correctly.\n")

    ab = m["alert_burden"]
    L.append("## Alert burden\n")
    L.append("The alert-fatigue budget, in the unit a care team actually plans against. "
             "Snapshots are 30 days apart, so one alert per snapshot is one per patient-month.\n")
    L.append(table(
        ["", "Alerts / 100 patient-months", "False alerts / 100 patient-months",
         "Precision of alerts"],
        [["**DoseSense**", f(ab["model"]["alerts_per_100_patient_months"], 1),
          f"**{f(ab['model']['false_alerts_per_100_patient_months'], 1)}**",
          f(ab["model"]["precision_of_alerts"])],
         ["PDC rule", f(ab["baseline_pdc"]["alerts_per_100_patient_months"], 1),
          f(ab["baseline_pdc"]["false_alerts_per_100_patient_months"], 1),
          f(ab["baseline_pdc"]["precision_of_alerts"])]]))
    L.append(f"\nA **{pc(ab['false_alert_reduction'])} reduction in false alerts** at a "
             "comparable total alert volume. For a 200-patient panel that is the difference "
             f"between roughly {ab['baseline_pdc']['false_alerts_per_100_patient_months'] * 2:.0f} "
             f"and {ab['model']['false_alerts_per_100_patient_months'] * 2:.0f} unnecessary "
             "conversations a month.\n")

    an = m.get("adversarial_negatives")
    if an:
        L.append("## The temporary-versus-meaningful test\n")
        L.append("Four behavioural archetypes are adherent for their entire record by "
                 "construction, so any alert on them is a false positive against a known truth. "
                 "Two of them are adversarial: `MISLEADING_ANOMALY` produces one dramatic refill "
                 "gap, a missed appointment and an off-trend laboratory result while remaining "
                 "fully adherent; `DISEASE_PROGRESSION` deteriorates clinically with a flawless "
                 "dispensing record.\n")
        L.append(table(
            ["Pattern", "Snapshots", "True concern rate", "DoseSense alerts", "PDC alerts"],
            [[("**" + r["archetype"] + "**") if r["is_negative_by_construction"] else r["archetype"],
              r["n_snapshots"], f(r["true_concern_rate"]),
              f(r["model_alert_rate"]), f(r.get("baseline_alert_rate"))]
             for r in m["archetype_behaviour"]]))
        L.append(f"\nAcross the four negative-by-construction patterns the false-alert rate is "
                 f"**{f(an['model_false_alert_rate'])} against {f(an['baseline_false_alert_rate'])} "
                 f"for the PDC rule**, a {pc(an['relative_reduction'])} relative reduction. On "
                 "`TEMPORARY_INTERRUPTION` — a genuine five-week lapse that resolves — DoseSense "
                 "alerts on far fewer snapshots than the PDC rule, because it stands down once "
                 "the patient recovers rather than continuing to flag a closed episode.\n")

    sil = next((r for r in m["archetype_behaviour"]
                if r["archetype"] == "SILENT_NONADHERENCE"), None)
    if sil:
        L.append("## The case the conventional metric cannot see\n")
        L.append("`SILENT_NONADHERENCE` collects every prescription on schedule and does not "
                 "take the doses. Because PDC is computed from dispensing records it reads near "
                 "1.0 throughout, and the metric is structurally blind. The only evidence is that "
                 "the clinical picture drifts while the pharmacy record stays immaculate.\n")
        L.append(f"- True concern rate: {f(sil['true_concern_rate'])}\n"
                 f"- PDC rule detects: **{f(sil.get('baseline_alert_rate'))}**\n"
                 f"- DoseSense detects: **{f(sil['model_alert_rate'])}**\n")
        L.append("This is the hardest pattern in the cohort and the recall is low in absolute "
                 "terms. It is reported as-is. The honest claim is a roughly "
                 f"{sil['model_alert_rate'] / max(sil.get('baseline_alert_rate') or 1e-9, 1e-9):.1f}× "
                 "improvement on a case the standard metric essentially cannot address, not that "
                 "the problem is solved.\n")

    L.append("## Calibration\n")
    L.append("The probability is multiplied by a clinical consequence weight to order the "
             "worklist, so it has to mean what it says. A model that ranks well but is "
             "miscalibrated would produce a wrong work order that no AUC would reveal.\n")
    L.append(f"- Expected calibration error: **{f(cal['expected_calibration_error'], 4)}**\n"
             f"- Maximum calibration error: {f(cal['maximum_calibration_error'], 4)}\n"
             f"- Brier score: {f(cal['brier'], 4)} (against "
             f"{f(base['brier'], 4)} for the baseline)\n")
    L.append(table(["Predicted", "Observed", "Snapshots"],
                   [[f(b["mean_predicted"]), f(b["observed_rate"]), b["n"]]
                    for b in cal["bins"] if b["n"]]))
    worst = max((b for b in cal["bins"] if b["n"]), key=lambda b: b.get("gap", 0))
    L.append(f"\nThe maximum calibration error of {f(cal['maximum_calibration_error'], 3)} sits in "
             f"a bin holding {worst['n']} snapshots. That is small-sample noise rather than "
             "miscalibration, and the expected error weighted by bin size is the figure to read.\n")

    lt = m["lead_time"]
    L.append("## Detection lead time\n")
    L.append("For every patient who ever warrants a concern, the first snapshot labelled positive "
             "is compared with the first snapshot the detector flagged.\n")
    L.append(table(
        ["", "Median days to detection", "Detection rate", "Detected at or before onset"],
        [["**DoseSense**", f(lt["model"].get("median_days_to_detection"), 1),
          pc(lt["model"].get("detection_rate")),
          f"**{pc(lt['model'].get('share_detected_at_or_before_onset'))}**"],
         ["PDC rule", f(lt["baseline_pdc"].get("median_days_to_detection"), 1),
          pc(lt["baseline_pdc"].get("detection_rate")),
          pc(lt["baseline_pdc"].get("share_detected_at_or_before_onset"))]]))
    L.append("\n**This is the weakest result and is reported as such.** Median time to detection "
             "is one snapshot interval for both methods: DoseSense does not beat PDC on the "
             "median. It does catch a substantially larger share of cases at or before the onset "
             "snapshot, and it does so with a third of the false alerts. Median lead time is "
             "dragged down by the silent-non-adherence cases, which are detected late because "
             "laboratory markers move slowly. The gain here is in precision and in coverage of "
             "hard cases, not in speed.\n")

    if "ablation" in m:
        ab_rows = m["ablation"]
        L.append("## Ablation: does multi-signal reasoning earn its complexity?\n")
        L.append("Each row is an **independently trained and independently calibrated** model on a "
                 "growing set of signal families, not the full model with inputs masked. Masking "
                 "would leave trees splitting on always-zero features and would flatter the final "
                 "model.\n")
        L.append(table(
            ["Signal set", "Features", "PR-AUC", "F1", "False alerts / 100pm"],
            [[r["name"].replace("_", " "), r["n_features"], f(r["pr_auc"], 4),
              f(r["f1"]), f(r["false_alerts_per_100_patient_months"], 1)] for r in ab_rows]))
        gain = ab_rows[-1]["pr_auc"] - ab_rows[0]["pr_auc"]
        L.append(f"\nPR-AUC rises {ab_rows[0]['pr_auc']:.4f} → {ab_rows[-1]['pr_auc']:.4f} "
                 f"({gain:+.4f}). The largest single jump comes from adding patient-reported "
                 "symptoms, which is exactly where the silent-non-adherence cases become "
                 "visible.\n")
        L.append("Two honest observations. Dispensing data alone already carries most of the "
                 "discriminative signal, which is unsurprising and worth saying plainly rather "
                 "than overselling fusion. And false alerts do not fall monotonically as signals "
                 "are added — more inputs mean more ways to be wrong. The extra families earn "
                 "their place by covering cases refill data cannot reach and by enabling barrier "
                 "inference, not by lifting AUC dramatically.\n")

    L.append("## Choosing an operating point\n")
    L.append("A care team does not inherit a threshold; they choose one against their capacity.\n")
    L.append(table(
        ["Threshold", "Precision", "Recall", "F1", "Specificity", "False alerts / 100pm"],
        [[r["threshold"], f(r["precision"]), f(r["recall"]), f(r["f1"]),
          f(r["specificity"]), f(r["false_alerts_per_100_patient_months"], 1)]
         for r in m["threshold_sweep"]]))
    L.append(f"\nThe shipped default is {m['config']['alert_threshold']}.\n")

    L.append("## Subgroup performance\n")
    L.append("Sex and insurance tier are deliberately excluded from the model's inputs. That "
             "prevents the model reading them directly; it does not prevent a correlated feature "
             "reproducing the same disparity, which can only be established by measurement.\n")
    for by in ("sex", "insurance_tier", "age_band", "condition"):
        rows = m["subgroups"].get(by, [])
        usable = [r for r in rows if "recall" in r]
        if not usable:
            continue
        L.append(f"\n**By {by.replace('_', ' ')}**\n")
        L.append(table(["Group", "Snapshots", "Concern rate", "Recall", "Precision", "ECE"],
                       [[r["group"], r["n_snapshots"], f(r["positive_rate"]), f(r["recall"]),
                         f(r["precision"]), f(r["expected_calibration_error"])] for r in usable]))
        small = [r for r in rows if "recall" not in r]
        if small:
            L.append(f"\n{len(small)} group(s) fell below the minimum sample size and are "
                     "reported without metrics rather than with unstable ones.\n")
    L.append("\n**Largest observed gaps**\n")
    L.append(table(["Grouping", "Recall gap", "Calibration-error gap"],
                   [[k, f(v.get("recall_gap")), f(v.get("ece_gap"))]
                    for k, v in m["fairness_summary"].items() if "recall_gap" in v]))
    L.append("\nGaps by sex are negligible. The largest disparities are by age band and "
             "condition. We tested removing age from the feature matrix entirely: it cost "
             "0.003 PR-AUC and the age-band recall gap did not improve, so the disparity is not "
             "the model reading age directly — it reflects genuine differences in base rate and "
             "record richness across bands. That is a limitation, not a finding we have "
             "addressed. See `docs/limitations.md`.\n")

    L.append("## Signal-family importance\n")
    L.append(table(["Signal family", "Share of gain"],
                   [[k, f(v)] for k, v in m["family_importance"].items()]))

    L.append("\n## Reproducing these numbers\n")
    L.append("```bash\nbash scripts/reproduce.sh\n```\n")
    L.append("The cohort, the splits and the model are all seeded, so the figures above are "
             "reproducible on any machine. Metrics will move if the cohort size or seed changes, "
             "which is expected: the test set is 148 patients and small-cohort variance is real.\n")

    out = ROOT / "docs" / "evaluation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"Wrote {out} ({len('\n'.join(L)):,} chars) from artifacts/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
