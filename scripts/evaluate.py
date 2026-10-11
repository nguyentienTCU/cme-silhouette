"""Evaluate an existing checkpoint on the original validation split."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='A new directory for evaluation artifacts.')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    import numpy as np
    import torch
    from src.datasets import build_dataloaders
    from src.models import build_model
    from src.evaluator import validate, visualize_predictions
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    _, val_loader = build_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config['model_name']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    val_loss, val_iou = validate(model, val_loader, device, config['alpha_loss'])
    with (args.output_dir / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump({'val_loss': val_loss, 'val_iou': val_iou}, stream, indent=4)
    with (args.output_dir / 'config.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(config, evaluation_checkpoint=str(args.checkpoint)), stream, indent=4)
    visualize_predictions(model, val_loader, device, config['model_name'], args.output_dir, show=not args.no_show)


if __name__ == '__main__':
    main()
