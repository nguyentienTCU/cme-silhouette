"""Read-only project checks using only Python's standard library."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from src.configuration import load_config, output_path


def main():
    files = [ROOT / 'train_model.py', ROOT / 'mask_Creation.py']
    for directory in ('src', 'scripts', 'tests'):
        files.extend(sorted((ROOT / directory).glob('*.py')))
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        package = path.parent.name if path.parent != ROOT else ''
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.level:
                    module = importlib.util.resolve_name('.' * node.level + module, package)
                modules = [module]
            for module in modules:
                if module.split('.')[0] in ('src', 'scripts'):
                    target = ROOT.joinpath(*module.split('.'))
                    if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                        raise AssertionError(f'{path.relative_to(ROOT)}: missing local import {module}')
    print(f'PASS: syntax and local import targets ({len(files)} Python files)')

    notebook = json.loads((ROOT / 'DataAnalysis.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    config = load_config(ROOT / 'config.json')
    print('PASS: notebook JSON/code syntax and current JSON configuration')

    for name, category in [('manet_lr1e-5_loss_bce70_dice30', 'loss_weights'),
                           ('manet_lr_1e-5', 'learning_rate'),
                           ('manet_baseline', 'baseline'), ('custom_run', 'experiments')]:
        candidate = dict(config, run_name=name)
        expected = ROOT / 'outputs' / 'manet' / category / name
        assert output_path(candidate) == expected
    for name in ('manet', 'deeplabv3plus', 'upernet', 'fpn', 'linknet'):
        assert (ROOT / 'configs/models' / f'{name}.yaml').is_file()
    for name in ('manet', 'deeplabv3plus', 'upernet'):
        assert (ROOT / 'configs/optuna' / f'{name}.yaml').is_file()
    print('PASS: output path classification and configuration file locations')
    print('DEFERRED TO NOVA: YAML parsing, external imports, CPU pipeline tests, dataset access, CUDA, model execution.')
    print('Remote dataset CSV:', config['csv_file'])
    print('Remote dataset root:', config['root_dir'])


if __name__ == '__main__':
    main()
