"""Regression checks for misleading metric denominators and stale claims."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dosesense.evaluate import archetype_behaviour
from scripts.make_report import render_report, replace_block, START, END


def test_mixed_archetype_alert_rate_is_not_recall():
    # One true positive and two false positives: 3/4 alerts, but 1/2 recall.
    panel = pd.DataFrame({'archetype': ['mixed'] * 4, 'patient_id': ['a', 'a', 'b', 'b'],
                          'label': [1, 1, 0, 0]}, index=[10, 20, 30, 40])
    row = archetype_behaviour(panel, np.array([.9, .1, .9, .9]),
                             np.array([1, 1, 0, 0]))[0]
    assert row['model_alert_rate'] == .75
    assert row['model_recall'] == .5
    assert row['baseline_alert_rate'] == .5
    assert row['baseline_recall'] == 1
    assert row['model_true_positives'] == 1
    assert row['n_positive_snapshots'] == 2
    assert not row['is_negative_by_construction']


def test_no_positive_group_has_undefined_recall():
    panel = pd.DataFrame({'archetype': ['negative'] * 2, 'patient_id': ['a', 'b'],
                          'label': [0, 0]})
    row = archetype_behaviour(panel, np.array([.9, .1]), np.array([0, 0]))[0]
    assert row['is_negative_by_construction']
    assert row['model_recall'] is None
    assert row['baseline_recall'] is None
    assert row['model_alert_rate'] == .5


def test_report_does_not_turn_alert_rate_into_recall_or_claim_earlier_detection():
    root = Path(__file__).resolve().parents[1]
    m = copy.deepcopy(json.loads((root / 'artifacts/metrics.json').read_text()))
    sil = next(r for r in m['archetype_behaviour'] if r['archetype'] == 'SILENT_NONADHERENCE')
    sil.update(n_positive_snapshots=10, model_true_positives=3, model_recall=.3,
               baseline_true_positives=2, baseline_recall=.2,
               model_alert_rate=.8, baseline_alert_rate=.9)
    # Reverse the timing comparison; report must print values without a fixed win story.
    m['lead_time']['model']['share_detected_at_or_before_onset'] = .1
    m['lead_time']['baseline_pdc']['share_detected_at_or_before_onset'] = .9
    text = render_report(m)
    assert '3/10 = 30.0%' in text
    assert '2/10 = 20.0%' in text
    assert '80.0% (model), 90.0% (baseline)' in text
    assert '10.0%' in text and '90.0%' in text
    assert 'substantially larger share' not in text
    assert 'prospective early warning' in text


def test_generated_block_preserves_reviewed_narrative():
    text = f'Before\n{START}\nstale claims\n{END}\nAfter'
    result = replace_block(text, START, END, 'new measured claims')
    assert result == f'Before\n{START}\nnew measured claims\n{END}\nAfter'


def test_cohort_is_identical_across_python_hash_seeds():
    import os
    import subprocess
    import sys
    code = '''
import hashlib
from dosesense.datagen import generate_cohort
cohort = generate_cohort(n_patients=48, seed=123)
payload = ''.join(name + frame.to_csv(index=False) for name, frame in sorted(cohort.items()))
print(hashlib.sha256(payload.encode()).hexdigest())
'''
    root = Path(__file__).resolve().parents[1]
    fingerprints = [subprocess.check_output(
        [sys.executable, '-c', code], cwd=root,
        env=dict(os.environ, PYTHONHASHSEED=value), text=True).strip()
        for value in ('1', '999')]
    assert fingerprints[0] == fingerprints[1]
