from pathlib import Path
import logging
import csv
import json
from copy import deepcopy
import numpy as np
import torch
from .losses import combined_loss
from .metrics import iou_metric
from .evaluator import validate
from .configuration import validate_config, create_output_dir
from .datasets import build_dataloaders
from .models import build_model

def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved at epoch {epoch+1} with loss {loss:.4f}")

def train_one_epoch(model, dataloader, optimizer, device, alpha_loss):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_samples = 0

    for inputs, masks in dataloader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = combined_loss(outputs, masks, alpha=alpha_loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_iou += iou_metric(outputs, masks) * inputs.size(0)
        num_samples += inputs.size(0)

    epoch_loss = running_loss / num_samples
    epoch_iou = running_iou / num_samples

    return epoch_loss, epoch_iou


def fit(model, train_loader, val_loader, optimizer, device, config,
        output_dir, *, save_checkpoints=True, logger=None, trial=None):
    """Shared epoch loop. Checkpoints still follow strictly improving training loss."""
    best_loss = float('inf')
    best_train_epoch = None
    best_validation = None
    best_val_iou = float('-inf')
    best_val_iou_epoch = None
    history = []

    for epoch in range(config['num_epochs']):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, device, config['alpha_loss']
        )

        val_loss, val_iou = validate(
            model, val_loader, device, config['alpha_loss']
        )

        row = dict(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_iou=train_iou,
            val_loss=val_loss,
            val_iou=val_iou
        )

        history.append(row)

        logging.info(
            'Epoch %s/%s: Train Loss = %.4f, Train IoU = %.4f, Val Loss = %.4f, Val IoU = %.4f',
            epoch + 1, config['num_epochs'],
            train_loss, train_iou, val_loss, val_iou
        )

        # Log metrics to Weights & Biases
        if logger is not None:
            logger.log(row)

        # Save checkpoint based on minimum training loss
        if train_loss < best_loss:
            best_loss = train_loss
            best_train_epoch = row.copy()

            if save_checkpoints:
                save_checkpoint(
                    model, optimizer, epoch, train_loss,
                    output_dir / config['checkpoint_path']
                )

        # Track best validation loss
        if best_validation is None or val_loss < best_validation['val_loss']:
            best_validation = row.copy()

        # Track best validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_val_iou_epoch = epoch + 1

        if trial is not None:
            trial.report(val_iou, step=epoch)
            if trial.should_prune():
                from optuna import TrialPruned
                raise TrialPruned(f'Trial pruned at epoch {epoch+1}')

    metrics = {
        'final': history[-1],
        'best_validation': best_validation,
        'best_val_iou': best_val_iou,
        'best_val_iou_epoch': best_val_iou_epoch,
        'checkpoint_selection': 'minimum_train_loss',
        'best_training': best_train_epoch,
        'checkpoint_saved': save_checkpoints
    }

    with (output_dir / 'metrics.json').open('x', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=4)

    return metrics



def run_training(config, *, output_root=None, output_dir=None,
                 save_checkpoints=True, visualize=True, show=True, trial=None):
    """Run the existing pipeline with a caller-provided configuration."""
    import wandb
    from .evaluator import visualize_predictions

    config = deepcopy(validate_config(config))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        output_dir = create_output_dir(config, output_root)

    saved_config = dict(
        config,
        save_checkpoints=save_checkpoints,
        visualize_predictions=visualize,
        show_predictions=show
    )

    with (output_dir / 'config.json').open('x', encoding='utf-8') as stream:
        json.dump(saved_config, stream, indent=4)

    wandb.init(
        project='cme-segmentation',
        name=config['run_name'],
        config=saved_config
    )

    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

        train_loader, val_loader = build_dataloaders(config)

        wandb.config.update({
            'total_images': len(train_loader.dataset) + len(val_loader.dataset),
            'train_images': len(train_loader.dataset),
            'val_images': len(val_loader.dataset)
        })

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('Training on device: %s', device)

        model = build_model(config['model_name']).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate']
        )

        metrics = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            config,
            output_dir,
            save_checkpoints=save_checkpoints,
            logger=wandb,
            trial=trial
        )

        if visualize:
            visualize_predictions(
                model,
                val_loader,
                device,
                config['model_name'],
                output_dir,
                show=show
            )

        return {
            'output_dir': str(output_dir),
            'metrics': metrics
        }

    finally:
        wandb.finish()

