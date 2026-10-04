"""Aggregate independent experiment runs, keeping bootstrap uncertainty separate."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(records):
    if len(records) < 2:
        raise ValueError('At least two independent seed runs are required')
    seeds = [r['config']['seed'] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate experiment seeds')
    if len({r['test_ids_sha256'] for r in records}) != 1:
        raise ValueError('Test sets differ across runs')
    ignored = {'seed', 'metrics_save_path', 'llm_responses_save_path'}
    configs = [{k: v for k, v in r['config'].items() if k not in ignored} for r in records]
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError('Experiment settings differ across runs')
    summary = {}
    keys = {k for k, v in records[0]['metrics'].items() if isinstance(v, dict) and 'value' in v}
    if not keys:
        raise ValueError('No per-run metric values were found')
    for key in sorted(keys):
        values = [r['metrics'][key]['value'] for r in records]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'Non-finite values for {key}')
        summary[key] = {'mean': statistics.mean(values), 'std': statistics.stdev(values),
                        'values': values}
    return {'seeds': seeds, 'n_runs': len(records), 'std_ddof': 1,
            'test_ids_sha256': records[0]['test_ids_sha256'], 'metrics': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize([json.loads(p.read_text()) for p in args.runs]), indent=2))
