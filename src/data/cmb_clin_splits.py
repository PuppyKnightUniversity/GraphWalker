import hashlib
import json
import pickle
import random
from pathlib import Path

from data.splits import normalize_patient_id

SPLIT_NAMES = ('train', 'val', 'test')


def _case_groups(data, field):
    if field not in data:
        raise ValueError(f'CMB-Clin requires {field!r} on every question; provide the original clinical case ID')
    n = len(data[field])
    if not n or any(len(values) != n for values in data.values()):
        raise ValueError('CMB-Clin question fields must be nonempty and aligned')
    groups = {}
    for i, value in enumerate(data[field]):
        try:
            case = normalize_patient_id(value)
        except ValueError as error:
            raise ValueError(f'Missing or invalid clinical case ID on question row {i}') from error
        groups.setdefault(case, []).append(i)
    return groups, n


def _allocate_cases(groups, sizes, seed):
    cases = sorted(groups)
    random.Random(seed).shuffle(cases)
    states = {(0, 0): ()}
    processed = 0
    for case in cases:
        count = len(groups[case])
        processed += count
        next_states = {}
        for (train, val), assignment in states.items():
            for split in range(3):
                t = train + (count if split == 0 else 0)
                v = val + (count if split == 1 else 0)
                if t <= sizes[0] and v <= sizes[1] and processed - t - v <= sizes[2]:
                    next_states.setdefault((t, v), assignment + (split,))
        states = next_states
    assignment = states.get((sizes[0], sizes[1]))
    if assignment is None:
        raise ValueError(f'Cannot achieve question counts {tuple(sizes)} with intact clinical cases; '
                         'supply feasible --cmb_clin_split_sizes or correct the case IDs')
    return {name: sorted(case for case, split in zip(cases, assignment) if split == i)
            for i, name in enumerate(SPLIT_NAMES)}


def _validate_manifest(manifest, groups, sizes):
    if not isinstance(manifest, dict):
        raise ValueError('The case split manifest must be a JSON object')
    mapping = manifest.get('case_ids', manifest)
    if not isinstance(mapping, dict) or set(mapping) != set(SPLIT_NAMES):
        raise ValueError('The case split manifest must define train, val, and test case-ID lists')
    seen, normalized = set(), {}
    for name, size in zip(SPLIT_NAMES, sizes):
        if not isinstance(mapping[name], list):
            raise ValueError(f'Manifest {name} must be a list of case IDs')
        cases = [normalize_patient_id(value) for value in mapping[name]]
        if len(cases) != len(set(cases)) or seen.intersection(cases):
            raise ValueError('A clinical case cannot occur twice or in more than one split')
        if set(cases) - groups.keys():
            raise ValueError(f'Manifest {name} contains case IDs absent from the input')
        if sum(len(groups[case]) for case in cases) != size:
            raise ValueError(f'Manifest {name} does not match the requested question count {size}')
        seen.update(cases)
        normalized[name] = sorted(cases)
    if seen != set(groups):
        raise ValueError('The case split manifest must cover every input clinical case')
    return normalized


def split_cmb_clin(args, data, logger=None):
    field = getattr(args, 'clinical_case_id_field', 'case_id')
    groups, n = _case_groups(data, field)
    sizes = tuple(getattr(args, 'cmb_clin_split_sizes', (83, 42, 83)))
    if len(sizes) != 3 or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in sizes) or sum(sizes) != n:
        raise ValueError(f'--cmb_clin_split_sizes must be three positive counts summing to {n}; got {sizes}')
    seed = getattr(args, 'data_split_seed', 3407)
    manifest_path = getattr(args, 'cmb_clin_split_manifest', None)
    mapping = (json.loads(Path(manifest_path).read_text()) if manifest_path
               else _allocate_cases(groups, sizes, seed))
    assignment_seed = mapping.get('data_split_seed') if manifest_path and isinstance(mapping, dict) else seed
    mapping = _validate_manifest(mapping, groups, sizes)
    splits = []
    for name in SPLIT_NAMES:
        indices = sorted(i for case in mapping[name] for i in groups[case])
        splits.append({key: [values[i] for i in indices] for key, values in data.items()})
    digest = hashlib.sha256(json.dumps({'data': data, 'case_ids': mapping}, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()
    output = Path(args.mid_data_dump_path) / 'cmb_clin' / f'case_split_v1_{digest[:16]}'
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'protocol': 'clinical_case_v1', 'case_id_field': field, 'data_split_seed': assignment_seed,
                'question_counts': dict(zip(SPLIT_NAMES, sizes)), 'case_ids': mapping,
                'input_and_assignment_sha256': digest}
    (output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    for name, split in zip(SPLIT_NAMES, splits):
        with (output / f'cmb_clin_{name}.pkl').open('wb') as stream:
            pickle.dump(split, stream)
    if logger:
        logger.info(f'CMB-Clin case-disjoint question counts: {sizes}; manifest: {output / "split_manifest.json"}')
    return tuple(splits)


def main():
    import argparse
    from data.prepare_clinical_text_data import prepare_clinical_text_data_extract_from_raw
    parser = argparse.ArgumentParser(description='CMB-Clin splits with exact question counts and indivisible clinical cases.')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--mid_data_dump_path', default='./mid_data')
    parser.add_argument('--clinical_case_id_field', default='case_id')
    parser.add_argument('--data_split_seed', type=int, default=3407)
    parser.add_argument('--cmb_clin_split_sizes', type=int, nargs=3, default=[83, 42, 83])
    parser.add_argument('--cmb_clin_split_manifest')
    args = parser.parse_args()
    args.dataset = 'cmb_clin'
    data = prepare_clinical_text_data_extract_from_raw(args)
    splits = split_cmb_clin(args, data)
    print('Question counts:', dict(zip(SPLIT_NAMES, (len(split[args.clinical_case_id_field]) for split in splits))))
    print('Case split manifests saved under:', Path(args.mid_data_dump_path) / 'cmb_clin')


if __name__ == '__main__':
    main()
