from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.metrics import f1_score
from utils import llm_eval


def test_f1_uses_fixed_threshold_not_test_optimum():
    scores, labels = [.1, .2, .3, .4], [0, 0, 1, 1]
    metrics = llm_eval.evaluate_binary_model(scores, labels, verbose=False)
    assert metrics['F1 Score'] == 0
    assert metrics['Threshold'] == .5
    assert llm_eval.evaluate_binary_model(scores, labels, verbose=False, threshold=.25)['F1 Score'] == 1


def test_bootstrap_does_not_refit_threshold_or_change_global_rng():
    np.random.seed(123)
    before = np.random.get_state()
    metrics = llm_eval.bootstrap_metrics([.1, .2, .3, .4], [0, 0, 1, 1],
                                         n_bootstrap=30, random_state=2, threshold=.5)
    assert metrics['F1_Score']['mean'] == 0
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_los_metrics_return_to_caller(monkeypatch):
    expected = {'AUC_Macro': {'value': .8}}
    monkeypatch.setattr(llm_eval, 'evaluate_multilabel_model_with_bootstrap', lambda **kwargs: expected)
    scores = [dict(zip('ABCD', [.7, .1, .1, .1]))]
    result = llm_eval.llm_response_evaluation(SimpleNamespace(dataset='mimic3_los'),
                                             ['A'], scores, {'y': [0]}, logger=Mock())
    assert result == expected


@pytest.mark.parametrize('scores', [[None], [{'A': 1.}], [dict.fromkeys('ABCD', 0.)]])
def test_missing_los_probabilities_are_not_replaced_with_uniform(scores):
    with pytest.raises(ValueError, match='LOS option probabilities'):
        llm_eval.llm_response_evaluation(SimpleNamespace(dataset='mimic3_los'),
                                         ['A'], scores, {'y': [0]}, logger=Mock())
