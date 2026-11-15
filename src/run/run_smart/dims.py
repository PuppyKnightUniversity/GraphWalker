import numbers

import numpy as np


NUM_CLASS = {'mortality': 2, 'readmission': 2, 'los': 4}
DEMO_DIM = {'mimic3': 0, 'mimic4': 2, 'tjh': 2}
CONFIG_KEYS = ('smart_input_dim', 'smart_demo_dim', 'smart_num_class')


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f'--{name} must be an integer >= {minimum}')
    return int(value)


def resolve_smart_dims(args, splits):
    dims = set()
    for split_index, split in enumerate(splits):
        for sample_index, sample in enumerate(split['data_smart']):
            try:
                values = np.asarray(sample['x'])
            except ValueError as error:
                raise ValueError(f'Invalid SMART matrix in split {split_index}, sample {sample_index}') from error
            if values.ndim != 2 or not all(values.shape):
                raise ValueError(f'SMART x must be a nonempty 2D matrix; got {values.shape} '
                                 f'in split {split_index}, sample {sample_index}')
            dims.add(values.shape[1])
    if len(dims) != 1:
        raise ValueError(f'Inconsistent SMART feature dimensions across samples: {sorted(dims)}')
    inferred = int(dims.pop())
    explicit = getattr(args, 'smart_input_dim', None)
    if explicit is not None:
        explicit = _integer(explicit, 'smart_input_dim')
        if explicit != inferred:
            raise ValueError(f'--smart_input_dim={explicit}, but the data has {inferred} features')
    source, separator, task = args.dataset.partition('_')
    if not separator:
        raise ValueError(f'Unsupported SMART dataset: {args.dataset}')
    demo = getattr(args, 'smart_demo_dim', None)
    if demo is None:
        if source not in DEMO_DIM:
            raise ValueError(f'No demographic default for {source}; specify --smart_demo_dim')
        demo = DEMO_DIM[source]
    demo = _integer(demo, 'smart_demo_dim', minimum=0)
    if demo > inferred:
        raise ValueError(f'--smart_demo_dim={demo} exceeds the {inferred} input features')
    classes = getattr(args, 'smart_num_class', None)
    expected_classes = NUM_CLASS.get(task)
    if classes is None:
        if expected_classes is None:
            raise ValueError(f'No class-count default for {task}; specify --smart_num_class')
        classes = expected_classes
    classes = _integer(classes, 'smart_num_class')
    if expected_classes is not None and classes != expected_classes:
        raise ValueError(f'--smart_num_class={classes}, but {task} requires {expected_classes} classes')
    args.smart_input_dim = inferred
    args.smart_demo_dim = demo
    args.smart_num_class = classes
    if getattr(args, 'smart_max_len', None) is None:
        args.smart_max_len = args.period_length
    return args


def smart_checkpoint_config(args):
    return {key: getattr(args, key) for key in CONFIG_KEYS}


def validate_smart_checkpoint(args, checkpoint, *, load_classifier=False):
    saved = checkpoint.get('config')
    if saved is None:
        return
    if not isinstance(saved, dict):
        raise ValueError('Checkpoint config must be a dictionary')
    keys = CONFIG_KEYS if load_classifier else CONFIG_KEYS[:2]
    for key in keys:
        if key in saved and saved[key] != getattr(args, key, None):
            raise ValueError(f'Checkpoint was trained with {key}={saved[key]}, '
                             f'but the current configuration uses {getattr(args, key, None)}')
