#!/usr/bin/env python3
"""Generate evaluation tables and marked submission claims from saved metrics.

Run after ml/evaluate.py. --check exits nonzero if generated content is stale.
Narrative outside the marked blocks still requires human review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN GENERATED RESULTS -->'
END = '<!-- END GENERATED RESULTS -->'
PITCH_START = '<!-- BEGIN GENERATED PITCH RESULTS -->'
PITCH_END = '<!-- END GENERATED PITCH RESULTS -->'


def f(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def pc(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join('---' for _ in headers) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def results_summary(m):
    model, baseline = m['model'], m['baseline_pdc']
    ds, burden = m['dataset'], m['alert_burden']
    return '\n\n'.join([
        f"**Synthetic-data evaluation only.** {ds['n_test_patients']} held-out patients, "
        f"{ds['n_test_snapshots']:,} snapshots; seed {m['model_metadata']['seed']}. "
        f"Generated from `artifacts/metrics.json` ({m['generated_at']}).",
        table(['Metric', 'DoseSense', 'PDC / refill-gap baseline'], [
            ['ROC-AUC', f(model['roc_auc']), f(baseline['roc_auc'])],
            ['PR-AUC', f(model['pr_auc']), f(baseline['pr_auc'])],
            ['Precision', f(model['precision']), f(baseline['precision'])],
            ['Recall', f(model['recall']), f(baseline['recall'])],
            ['F1', f(model['f1']), f(baseline['f1'])],
            ['False alerts / 100 patient-months',
             f(burden['model']['false_alerts_per_100_patient_months'], 2),
             f(burden['baseline_pdc']['false_alerts_per_100_patient_months'], 2)],
        ]),
        f"Relative change in false alerts: {pc(burden['false_alert_reduction'])} reduction "
        "(a negative value means an increase). "
        f"Expected calibration error: {f(m['calibration']['expected_calibration_error'], 4)}. "
        f"The evaluated model uses {m['model_metadata']['n_features']} features.",
        'These are model-threshold results. The dashboard additionally applies uncertainty, '
        'state and priority rules. These measurements do not establish clinical outcomes, '
        'real-world accuracy or end-to-end workflow savings.'
    ])


def pitch_results(m):
    return (
        f"On {m['dataset']['n_test_patients']} held-out synthetic patients, DoseSense achieved "
        f"a PR-AUC of {f(m['model']['pr_auc'])}, compared with "
        f"{f(m['baseline_pdc']['pr_auc'])} for our refill baseline. "
        f"The relative reduction in model-threshold false alerts was "
        f"{pc(m['alert_burden']['false_alert_reduction'])}. "
        "These are simulator results, not clinical validation."
    )


def render_report(m):
    L = ['# Evaluation', results_summary(m)]
    L += ['## Protocol',
          'Patients are disjoint across training, calibration and test sets. Windows use only '
          'records available at the snapshot date. The comparator combines PDC with a refill-gap '
          'rule. No real patient records were used.',
          f"Cohort: {m['dataset']['n_patients']} patients and "
          f"{m['dataset']['n_snapshots']:,} snapshots. Snapshot spacing: "
          f"{m['config']['snapshot_interval_days']} days. Model threshold: "
          f"{m['config']['alert_threshold']}. The label uses a trailing "
          f"{m['config']['label_window_days']}-day window and a minimum non-adherent fraction of "
          f"{m['config']['label_min_nonadherent_fraction']}.",
          '## Alert burden',
          'One threshold crossing per snapshot counts as an alert, including repeated alerts '
          'on the same patient. This is not a measured count of calls made or avoided.',
          table(['Method', 'Alerts / 100 patient-months', 'False alerts / 100 patient-months'], [
              [name, f(m['alert_burden'][key]['alerts_per_100_patient_months'], 2),
               f(m['alert_burden'][key]['false_alerts_per_100_patient_months'], 2)]
              for name, key in [('DoseSense', 'model'), ('PDC rule', 'baseline_pdc')]
          ]),
          '## Behaviour by pattern',
          '**Alert rate** divides all flagged snapshots by all snapshots in that pattern. '
          '**Recall** divides true-positive flags by positive snapshots only. A pattern can '
          'include both positive and negative snapshots; its alert rate is not its recall. '
          'Recall is undefined (—) when there are no positive snapshots.',
          table(['Pattern', 'Patients', 'Snapshots', 'True concern rate', 'Model alert rate',
                 'PDC alert rate', 'Model recall', 'PDC recall'], [
              [r['archetype'], r['n_patients'], r['n_snapshots'], f(r['true_concern_rate']),
               f(r['model_alert_rate']), f(r.get('baseline_alert_rate')),
               f(r.get('model_recall')), f(r.get('baseline_recall'))]
              for r in m['archetype_behaviour']])]
    neg = [r for r in m['archetype_behaviour'] if r['is_negative_by_construction']]
    if neg and m.get('adversarial_negatives'):
        an = m['adversarial_negatives']
        L.append(f"{len(neg)} patterns contain no positive labels in this test set: " +
                 ', '.join(r['archetype'] for r in neg) + '. ' +
                 f"Their pooled false-alert rates are {f(an['model_false_alert_rate'])} "
                 f"(DoseSense) and {f(an['baseline_false_alert_rate'])} (PDC). "
                 'The number of all-negative groups is determined from labels, not assumed.')
    sil = next((r for r in m['archetype_behaviour'] if r['archetype'] == 'SILENT_NONADHERENCE'), None)
    if sil:
        L += ['## Silent non-adherence',
              'This simulated pattern decouples medication collection from the hidden adherence '
              'state. It tests whether other records can contribute evidence when refills look normal.',
              f"Alert rates across all {sil['n_snapshots']} snapshots: "
              f"{pc(sil['model_alert_rate'])} (model), {pc(sil.get('baseline_alert_rate'))} (baseline)."]
        if 'n_positive_snapshots' in sil:
            L.append(f"Recall among {sil['n_positive_snapshots']} positive snapshots: "
                     f"{sil['model_true_positives']}/{sil['n_positive_snapshots']} = "
                     f"{pc(sil['model_recall'])} (model), "
                     f"{sil['baseline_true_positives']}/{sil['n_positive_snapshots']} = "
                     f"{pc(sil['baseline_recall'])} (baseline). "
                     'These are repeated observations of a small patient subgroup, not '
                     'independent clinical cases. Higher alert volume alone does not prove better detection.')
        else:
            L.append('Recall was not saved in this artifact. Rerun evaluation to calculate it; '
                     'it cannot be recovered from marginal alert rates alone.')
    cal = m['calibration']
    L += ['## Calibration',
          f"ECE: {f(cal['expected_calibration_error'], 4)}; maximum bin error: "
          f"{f(cal['maximum_calibration_error'], 4)}; Brier score: {f(cal['brier'], 4)}.",
          table(['Mean prediction', 'Observed rate', 'Snapshots'],
                [[f(b['mean_predicted']), f(b['observed_rate']), b['n']]
                 for b in cal['bins'] if b['n']]),
          'ECE is a bin-weighted average, not a bound on each prediction. Small bins are '
          'unstable; their errors cannot simply be dismissed as noise. This calibration '
          'has not been tested on clinical data.']
    lt = m['lead_time']
    L += ['## Detection timing',
          'The current calculation selects the first flagged snapshot at or after onset; '
          'if none exists, it uses the last pre-onset flag. Consequently, a pre-onset flag '
          'does not take precedence over a later flag. Timing statistics are conditional '
          'on patients with a selected flag; never-flagged patients are reported separately. '
          'This retrospective convention is not evidence of prospective early warning.',
          table(['Method', 'Onset patients', 'Never flagged', 'Median days',
                 'Any flag / onset patients', 'At/before onset / flagged patients'], [
              [name, lt[key]['n_with_onset'], lt[key]['n_never_detected'],
               f(lt[key].get('median_days_to_detection'), 1), pc(lt[key].get('detection_rate')),
               pc(lt[key].get('share_detected_at_or_before_onset'))]
              for name, key in [('DoseSense', 'model'), ('PDC rule', 'baseline_pdc')]
          ])]
    if m.get('ablation'):
        rows = m['ablation']
        L += ['## Ablation',
              'Each row is independently trained and calibrated on the same patient splits. '
              'The experimental row is excluded from the shipped feature set.',
              table(['Signal set', 'Features', 'PR-AUC', 'F1', 'False alerts / 100pm'],
                    [[r['name'], r['n_features'], f(r['pr_auc'], 4), f(r['f1']),
                      f(r['false_alerts_per_100_patient_months'], 2)] for r in rows])]
        refill = next(r for r in rows if r['name'] == 'refill_only')
        full = next(r for r in rows if r['name'] == 'all_signals')
        L.append(f"Shipped-family ablation versus refill-only: "
                 f"{refill['pr_auc']:.4f} → {full['pr_auc']:.4f} "
                 f"({full['pr_auc'] - refill['pr_auc']:+.4f} PR-AUC). "
                 'Intermediate additions need not improve performance monotonically. '
                 'The separately trained ablation is not the headline ensemble.')
    L += ['## Operating points',
          'This is a descriptive test-set sweep. Choose thresholds on development data '
          'and evaluate on new held-out data; do not select a winner from this table and '
          'treat its test performance as an unbiased estimate.',
          table(['Threshold', 'Precision', 'Recall', 'F1', 'Specificity', 'False alerts / 100pm'],
                [[r['threshold'], f(r['precision']), f(r['recall']), f(r['f1']),
                  f(r['specificity']), f(r['false_alerts_per_100_patient_months'], 2)]
                 for r in m['threshold_sweep']]),
          '## Subgroups',
          'Excluding sex and insurance tier from features does not eliminate proxy effects '
          'or establish fairness. These synthetic subgroup measurements do not explain causation.']
    for group, rows in m['subgroups'].items():
        L += [f'### {group}', table(['Group', 'Snapshots', 'Recall', 'Precision', 'ECE'],
              [[r['group'], r['n_snapshots'], f(r.get('recall')), f(r.get('precision')),
                f(r.get('expected_calibration_error'))] for r in rows])]
    L.append(table(['Grouping', 'Recall gap', 'ECE gap'],
                   [[group, f(r.get('recall_gap')), f(r.get('ece_gap'))]
                    for group, r in m['fairness_summary'].items()]))
    if m.get('richness_audit'):
        L += ['## Approximate data-richness audit',
              'This audit uses fixed ensemble disagreement (0.02), simplified family counts '
              'and no change-detection context. It is not the serving abstention policy '
              'evaluated end to end. Recall among committed cases excludes abstentions '
              'and must not be interpreted as recall across all patients.',
              table(['Stratum', 'Snapshots', 'Abstention rate', 'Committed recall'],
                    [[r['stratum'], r['n_snapshots'], pc(r['abstention_rate']),
                      f(r.get('committed_recall'))] for r in m['richness_audit']])]
    L += ['## Feature importance',
          table(['Family', 'Share of gain'], [[k, f(v)] for k, v in m['family_importance'].items()]),
          'Importance describes how this ensemble uses simulated features; it is not causal evidence.',
          '## Reproduction',
          '```bash\nbash scripts/reproduce.sh\npython scripts/make_report.py --check\n```',
          'Seeds control cohort generation and splits. Library versions can change results; '
          'use requirements-lock.txt for the verified environment. Reproducing simulator '
          'results does not validate the simulator or a clinical deployment.']
    if m.get('environment'):
        L.append('```json\n' + json.dumps(m['environment'], indent=2) + '\n```')
    return '\n\n'.join(L) + '\n'


def replace_block(text, start, end, content):
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f'Expected one generated block: {start}')
    a = text.index(start) + len(start)
    b = text.index(end, a)
    return text[:a] + '\n' + content + '\n' + text[b:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    m = json.loads((ROOT / 'artifacts/metrics.json').read_text())
    outputs = {ROOT / 'docs/evaluation.md': render_report(m)}
    for relative in ('README.md', 'docs/presentation.md'):
        path = ROOT / relative
        content = replace_block(path.read_text(), START, END, results_summary(m))
        if PITCH_START in content:
            content = replace_block(content, PITCH_START, PITCH_END, pitch_results(m))
        outputs[path] = content
    stale = []
    for path, content in outputs.items():
        if path.read_text() == content:
            continue
        stale.append(str(path.relative_to(ROOT)))
        if not args.check:
            path.write_text(content)
    if args.check and stale:
        print('Stale generated content: ' + ', '.join(stale), file=sys.stderr)
        return 1
    print('Generated content is current.' if args.check else 'Updated evaluation and submission claims.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
