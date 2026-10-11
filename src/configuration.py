"""JSON/YAML configuration and collision-safe experiment paths."""
import json
from pathlib import Path
from .models import MODEL_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('run_name', 'csv_file', 'root_dir', 'target_size', 'batch_size',
            'num_workers', 'num_epochs', 'learning_rate', 'checkpoint_path',
            'model_name', 'alpha_loss', 'seed')
EXPERIMENT_TYPES = ('baseline', 'learning_rate', 'loss_weights', 'experiments')


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping.')
    missing = [key for key in REQUIRED if config.get(key) is None]
    if missing:
        raise ValueError('Missing configuration values: ' + ', '.join(missing))
    if config['model_name'] not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {config['model_name']}")
    for key in ('run_name', 'checkpoint_path'):
        value = config[key]
        if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\:'):
            raise ValueError(f'{key} must be a single filename or directory name.')
    if config['num_epochs'] < 1 or config['batch_size'] < 1 or config['num_workers'] < 0:
        raise ValueError('Epochs/batch size must be positive and workers nonnegative.')
    return config


def load_config(path):
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        if path.suffix.lower() in ('.yaml', '.yml'):
            import yaml
            config = yaml.safe_load(stream)
        else:
            config = json.load(stream)
    return validate_config(config)


def experiment_type(config):
    explicit = config.get('experiment_type')
    if explicit is not None:
        if explicit not in EXPERIMENT_TYPES:
            raise ValueError(f'Unknown experiment_type: {explicit}')
        return explicit
    name = config['run_name']
    if '_loss_' in name:
        return 'loss_weights'
    if '_lr' in name:
        return 'learning_rate'
    if name.endswith('_baseline'):
        return 'baseline'
    return 'experiments'


def output_path(config, output_root=None):
    validate_config(config)
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / 'outputs'
    return root / config['model_name'] / experiment_type(config) / config['run_name']


def create_output_dir(config, output_root=None):
    path = output_path(config, output_root)
    # Never silently resume or overwrite an existing experiment.
    path.mkdir(parents=True, exist_ok=False)
    return path
