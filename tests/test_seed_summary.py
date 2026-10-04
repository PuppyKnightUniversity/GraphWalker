import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'src/scripts/graph_walker/summarize_seed_runs.py'
spec = importlib.util.spec_from_file_location('seed_summary', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(seed, value, fingerprint='fixed'):
    return {'config': {'seed': seed, 'data_split_seed': 3407},
            'test_ids_sha256': fingerprint,
            'metrics': {'AUROC': {'value': value, 'bootstrap_std': .9}}}


def test_summary_uses_independent_run_values():
    result = module.summarize([record(1, .7), record(2, .8), record(3, .9)])
    assert result['metrics']['AUROC']['mean'] == pytest.approx(.8)
    assert result['metrics']['AUROC']['std'] == pytest.approx(.1)
    assert result['std_ddof'] == 1


@pytest.mark.parametrize('records', [
    [record(1, .7), record(1, .8)],
    [record(1, .7), record(2, .8, fingerprint='changed')],
])
def test_summary_rejects_incomparable_runs(records):
    with pytest.raises(ValueError):
        module.summarize(records)
