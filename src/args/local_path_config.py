# Define mapping from dataset to dataset path
# NOTE: Please configure these paths according to your local environment
DATASET_PATH_MAP = {
    "mimic3_mortality": '<YOUR_DATA_PATH>/mid_data/processed_data/in-hospital-mortality',
    "mimic3_los": '<YOUR_DATA_PATH>/mid_data/processed_data/length-of-stay',
    "mimic4_mortality": '<YOUR_DATA_PATH>/reference_code/mimic4-data-processor/my_datasets/mimic-iv',
    "mimic4_los": '<YOUR_DATA_PATH>/reference_code/mimic4-data-processor/my_datasets/mimic-iv',
    "mimic4_readmission": '<YOUR_DATA_PATH>/reference_code/mimic4-data-processor/my_datasets/mimic-iv',
    "cmb_exam_patient": '',
    "tjh_los": '<YOUR_DATA_PATH>/reference_code/tjh',
    "tjh_mortality": '<YOUR_DATA_PATH>/reference_code/tjh',
    # NOTE: add more datasets here!
}

LLM_PATH_MAP = {
    'qwen2-5-7b-instruct': '',
    'qwen2-5-14b-instruct': '',
    'qwen3-14b-instruct': '<YOUR_LLM_PATH>/Qwen3-14B',
    'qwen3-32b-instruct': '<YOUR_LLM_PATH>/Qwen3-32B',
    'llama-3.1-8b-instruct': '<YOUR_LLM_PATH>/Meta-Llama-3.1-8B-Instruct',
    'ministral-3-14b-instruct': '',
    'qwen3-embedding-8b': '<YOUR_LLM_PATH>/Qwen3-Embedding-8B',
    'deepseek-r1-8b': '',
    'qwen2-5-72b-instruct': None,
    'deepseek-r1': None,
    'xiaobei-32b': None,
    'gpt-3.5-turbo': None,
    'gpt-5': None,
    # NOTE: add more models here!
}

EMBEDDING_MODEL_PATH_MAP = {
    'qwen3-embedding-8b': '<YOUR_LLM_PATH>/Qwen3-Embedding-8B',
}