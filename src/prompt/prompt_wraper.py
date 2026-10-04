"""EHR prompt entry points."""
from prompt.EHR_prompt.prompt_wraper import train_val_test_dataset_prompt_wrapper
from prompt.EHR_prompt.mimic3.utils import mimic3_smooth_hourly_data
from prompt.EHR_prompt.mimic3.mortality.prompt import (
    mimic3_mortality_prompt_wrapper, transform_mimic3_mortality_ehr_to_detail_prompt,
)
