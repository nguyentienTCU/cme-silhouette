"""Train from JSON or YAML without executing work on import."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import PROJECT_ROOT, load_config, output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'config.json')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--experiment-type', choices=('baseline', 'learning_rate', 'loss_weights', 'experiments'))
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--no-visualizations', action='store_true')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration and print output path; do not train or write files.')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.experiment_type:
        config['experiment_type'] = args.experiment_type
    if args.dry_run:
        print(json.dumps({'config': config, 'output_dir': str(output_path(config, args.output_root))}, indent=2))
        return
    from src.trainer import run_training
    run_training(config, output_root=args.output_root, save_checkpoints=not args.no_checkpoint,
                 visualize=not args.no_visualizations, show=not args.no_show)


if __name__ == '__main__':
    main()
