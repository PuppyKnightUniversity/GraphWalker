import json
import pickle
from types import SimpleNamespace

import pytest

from data.cmb_clin_splits import split_cmb_clin
from data.prepare_clinical_text_data import (
    prepare_clinical_text_data_extract_from_raw,
    prepare_clinical_text_data_train_val_test_split,
)


def args_for(tmp_path, **overrides):
    args = dict(dataset='cmb_clin', dataset_path=str(tmp_path),
                mid_data_dump_path=str(tmp_path / 'cache'), data_split_seed=3407, seed=1,
                train_ratio=.8, clinical_case_id_field='case_id',
                cmb_clin_split_sizes=[83, 42, 83], cmb_clin_split_manifest=None)
    return SimpleNamespace(**dict(args, **overrides))


def paper_sized_questions():
    # Synthetic 74 cases / 208 questions, with unequal case sizes.
    rows = [(f'case-{case:03}', f'q-{case}-{question}')
            for case in range(74) for question in range(3 if case < 60 else 2)]
    return {'case_id': [row[0] for row in rows], 'question_id': [row[1] for row in rows],
            'question': [f'Question {row[1]}' for row in rows]}


def memberships(splits):
    return [set(split['case_id']) for split in splits]


def test_paper_counts_with_all_cases_intact_and_manifest_roundtrip(tmp_path):
    args, data = args_for(tmp_path), paper_sized_questions()
    splits = prepare_clinical_text_data_train_val_test_split(args, data)
    assert [len(s['question_id']) for s in splits] == [83, 42, 83]
    groups = memberships(splits)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(set.union(*groups)) == 74
    all_questions = [q for split in splits for q in split['question_id']]
    assert len(all_questions) == len(set(all_questions)) == 208
    manifest = next((tmp_path / 'cache').rglob('split_manifest.json'))
    stored = json.loads(manifest.read_text())
    assert stored['question_counts'] == {'train': 83, 'val': 42, 'test': 83}
    args.cmb_clin_split_manifest = str(manifest)
    args.data_split_seed = 999
    assert memberships(split_cmb_clin(args, data)) == groups
    assert json.loads(manifest.read_text())['data_split_seed'] == 3407


def test_case_assignment_is_row_order_and_model_seed_independent(tmp_path):
    args, data = args_for(tmp_path), paper_sized_questions()
    expected = memberships(split_cmb_clin(args, data))
    args.seed = 999
    reverse = {key: list(reversed(values)) for key, values in data.items()}
    assert memberships(split_cmb_clin(args, reverse)) == expected


@pytest.mark.parametrize('bad_id', [None, '', 'null', True])
def test_missing_case_ids_fail_without_question_level_fallback(tmp_path, bad_id):
    data = paper_sized_questions()
    data['case_id'][0] = bad_id
    with pytest.raises(ValueError, match='clinical case ID'):
        split_cmb_clin(args_for(tmp_path), data)
    del data['case_id']
    with pytest.raises(ValueError, match='requires'):
        split_cmb_clin(args_for(tmp_path), data)


def test_custom_case_id_field(tmp_path):
    data = paper_sized_questions()
    data['original_case_id'] = data.pop('case_id')
    splits = split_cmb_clin(args_for(tmp_path, clinical_case_id_field='original_case_id'), data)
    assert [len(split['question_id']) for split in splits] == [83, 42, 83]


def test_impossible_counts_do_not_split_a_case(tmp_path):
    data = {'case_id': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']}
    with pytest.raises(ValueError, match='intact clinical cases'):
        split_cmb_clin(args_for(tmp_path, cmb_clin_split_sizes=[3, 2, 3]), data)


@pytest.mark.parametrize('sizes', [[83, 42, 82], [208, 0, 0], [True, 42, 165]])
def test_invalid_split_counts_rejected(tmp_path, sizes):
    with pytest.raises(ValueError, match='positive counts'):
        split_cmb_clin(args_for(tmp_path, cmb_clin_split_sizes=sizes), paper_sized_questions())


@pytest.mark.parametrize('change', ['overlap', 'unknown', 'missing', 'wrong_count'])
def test_invalid_manifests_rejected(tmp_path, change):
    args, data = args_for(tmp_path), paper_sized_questions()
    split_cmb_clin(args, data)
    path = next((tmp_path / 'cache').rglob('split_manifest.json'))
    manifest = json.loads(path.read_text())
    cases = manifest['case_ids']
    if change == 'overlap':
        cases['val'].append(cases['train'][0])
    elif change == 'unknown':
        cases['train'][0] = 'not-a-real-case'
    elif change == 'missing':
        cases.pop('test')
    else:
        cases['test'].append(cases['train'].pop())
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps(manifest))
    args.cmb_clin_split_manifest = str(bad)
    with pytest.raises(ValueError):
        split_cmb_clin(args, data)


def test_old_raw_and_row_split_caches_are_ignored(tmp_path):
    args, data = args_for(tmp_path), paper_sized_questions()
    raw_cache = tmp_path / 'cache' / 'cmb_clin'
    raw_cache.mkdir(parents=True)
    (raw_cache / 'cmb_clin_raw.pkl').write_bytes(pickle.dumps({'question': ['STALE']}))
    stale_split = raw_cache / 'seed1'
    stale_split.mkdir()
    for name in ('train', 'val', 'test'):
        (stale_split / f'cmb_clin_{name}.pkl').write_bytes(pickle.dumps({'question': ['STALE']}))
    records = [dict(zip(data, values)) for values in zip(*data.values())]
    (tmp_path / 'cmb_clin.jsonl').write_text('\n'.join(json.dumps(row) for row in records))
    current = prepare_clinical_text_data_extract_from_raw(args)
    assert current == data
    splits = prepare_clinical_text_data_train_val_test_split(args, current)
    assert [len(split['question']) for split in splits] == [83, 42, 83]
    # A different data file cannot be shadowed by the raw cache just written.
    records[0]['question'] = 'UPDATED'
    (tmp_path / 'cmb_clin.jsonl').write_text('\n'.join(json.dumps(row) for row in records))
    assert prepare_clinical_text_data_extract_from_raw(args)['question'][0] == 'UPDATED'
