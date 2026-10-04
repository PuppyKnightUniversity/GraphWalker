import importlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from args.ehrbase_args import parse_args
from data.longitudinal import prepare_longitudinal_data
from prompt.EHR_prompt.prompt_wraper import validate_detail_prompt_lengths
from run.runexp import runexp


def cli_args(monkeypatch, *extra):
    monkeypatch.setattr(sys, 'argv', ['main.py', '--dataset_path', '/data',
                                   '--llm_local_path', '/model', *extra])
    return parse_args()


@pytest.mark.parametrize('method,module_name,function', [
    ('llm_zero_shot', 'run.run_llm.run', 'run_llm_inference_for_ICL'),
    ('graph_walker', 'run.run_llm.run', 'run_llm_inference_for_ICL'),
    ('llm_sft_train', 'run.run_sft.run', 'run'),
    ('llm_sft_eval', 'run.run_sft.sft_eval', 'run'),
    ('llm_sft_eval_vllm', 'run.run_sft.sft_eval_vllm', 'run'),
])
def test_main_dispatch_and_metrics(monkeypatch, tmp_path, method, module_name, function):
    from data import prepare_ehr_data
    from utils import logger
    args = cli_args(monkeypatch, '--method', method, '--metrics_save_path', str(tmp_path / 'result.json'))
    datasets = tuple({'X': [1, 2], 'y': [0, 1], 'name': ['a', 'b']} for _ in range(3))
    monkeypatch.setattr(prepare_ehr_data, 'prepare_ehr_data', Mock(return_value=datasets))
    monkeypatch.setattr(logger, 'get_logger', Mock(return_value=Mock()))
    implementation = Mock(return_value={'score': 0.75})
    monkeypatch.setattr(importlib.import_module(module_name), function, implementation)
    assert runexp(args) == {'score': 0.75}
    assert implementation.call_args.args[:4] == (args, *datasets)
    result = json.loads(Path(args.metrics_save_path).read_text())
    assert result['test_size'] == 2 and result['metrics'] == {'score': 0.75}
    assert result['config']['method'] == method


def test_api_graph_scoring_fails_before_data_loading(monkeypatch):
    from data import prepare_ehr_data
    from utils import logger
    args = cli_args(monkeypatch, '--method', 'graph_walker', '--is_api')
    load = Mock()
    monkeypatch.setattr(prepare_ehr_data, 'prepare_ehr_data', load)
    monkeypatch.setattr(logger, 'get_logger', Mock(return_value=Mock()))
    with pytest.raises(ValueError, match='local token log-probabilities'):
        runexp(args)
    load.assert_not_called()


def test_tjh_cannot_be_split_into_an_in_domain_casebook():
    with pytest.raises(ValueError, match='training splits require a MIMIC dataset'):
        prepare_longitudinal_data(SimpleNamespace(dataset='tjh_mortality'))


def character_tokenizer(monkeypatch):
    monkeypatch.setitem(sys.modules, 'tiktoken', SimpleNamespace(
        get_encoding=lambda _: SimpleNamespace(encode=list)))


def test_length_validation_preserves_all_patients(monkeypatch):
    character_tokenizer(monkeypatch)
    dataset = {'detail': ['a', 'long record'], 'y': [0, 1], 'name': ['a', 'b']}
    before = pickle.dumps(dataset)
    with pytest.raises(ValueError, match='max_tokens_each_patient'):
        validate_detail_prompt_lengths(Mock(), dataset, max_tokens=3)
    assert pickle.dumps(dataset) == before
    validate_detail_prompt_lengths(Mock(), dataset, max_tokens=20)
    assert pickle.dumps(dataset) == before


@pytest.mark.parametrize('module_name', ['sft_eval', 'sft_eval_vllm'])
def test_sft_eval_rejects_overlength_records_without_dropping_patients(monkeypatch, module_name):
    character_tokenizer(monkeypatch)
    module = importlib.import_module('run.run_sft.' + module_name)
    dataset = {'detail': ['a', 'long record'], 'y': [0, 1]}
    monkeypatch.setattr(module, '_ensure_detail', lambda args, data: data)
    with pytest.raises(ValueError, match='max_tokens_each_patient'):
        module.run(SimpleNamespace(max_tokens_each_patient=3), {}, {}, dataset, Mock())
    assert dataset['y'] == [0, 1] and dataset['detail'] == ['a', 'long record']


@pytest.mark.parametrize('lora', [True, False])
def test_sft_launcher_preserves_configuration_from_any_directory(monkeypatch, tmp_path, lora):
    from run.run_sft import run as launcher
    args = cli_args(monkeypatch, '--method', 'llm_sft_train', '--toy_dataset',
                    '--toy_dataset_size_test', '13', '--ehr_records_path', '/records.jsonl',
                    '--data_split_seed', '7', '--sft_dry_run',
                    *([] if lora else ['--no-sft_use_lora']))
    assert args.sft_use_lora is lora
    monkeypatch.chdir(tmp_path)
    process = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(launcher.subprocess, 'run', process)
    result = launcher.run(args, {}, {}, {}, Mock())
    command = process.call_args.args[0]
    env = process.call_args.kwargs['env']
    assert command[command.index('--ehr_records_path') + 1] == '/records.jsonl'
    assert command[command.index('--data_split_seed') + 1] == '7'
    assert command[command.index('--toy_dataset_size_test') + 1] == '13'
    assert ('--sft_use_lora' if lora else '--no-sft_use_lora') in command
    assert '--sft_dry_run' in command
    assert Path(env['PYTHONPATH'].split(':')[0]) == Path(launcher.__file__).resolve().parents[2]
    assert result == {'returncode': 0, 'output_dir': args.sft_output_dir}
