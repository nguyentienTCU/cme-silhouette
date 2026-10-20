# CME Image Segmentation

Semantic segmentation of Coronal Mass Ejections (CMEs) in coronagraph images using five deep learning architectures: **MAnet, DeepLabV3Plus, UPerNet, FPN, and LinkNet**.

The project focuses on comparing segmentation performance across architectures, experimenting with training hyperparameters, and evaluating model predictions.

## Project Structure

```text
configs/
  models/             # Model training configurations
  optuna/             # Hyperparameter optimization templates

src/
  configuration.py    # Configuration and experiment management
  datasets.py         # Dataset preprocessing and loading
  models.py           # Model initialization
  losses.py           # BCE and Dice loss
  metrics.py          # Evaluation metrics
  trainer.py          # Training pipeline
  evaluator.py        # Model evaluation and visualization

scripts/
  train.py            # Run model training
  evaluate.py         # Evaluate trained checkpoints

tests/                # Unit tests
outputs/              # Experiment results
config.json           # Default training configuration
train_model.py        # Original training entry point
requirements.txt      # Project dependencies
```

## Setup

Training is performed on the Nova GPU server using the existing Python environment.

The required dependencies are listed in `requirements.txt`.

Dataset paths are specified in the training configuration. Update `csv_file` and `root_dir` if the dataset location changes.

## Training

Each model has a YAML configuration in `configs/models/`.

To train a model:

```bash
python scripts/train.py --config configs/models/manet.yaml --no-show
```

Replace `manet.yaml` with the configuration of another model as needed. Configurations must contain all required training settings.

To validate a configuration without starting training:

```bash
python scripts/train.py --config configs/models/manet.yaml --dry-run
```

The original training command is also supported:

```bash
python train_model.py
```

### Training Configuration

The models use an ImageNet-pretrained ResNet18 encoder with one input channel and one output class.

Training uses the Adam optimizer and a combined Binary Cross-Entropy (BCE) and Dice loss:

\[
L = \alpha L_{\text{BCE}} + (1-\alpha)L_{\text{Dice}}
\]

The `alpha_loss` parameter determines the weighting between BCE and Dice loss.

Segmentation performance is evaluated using Intersection over Union (IoU). Training and validation metrics are logged to **Weights & Biases (W&B)** after every epoch.

Model checkpoints are selected based on minimum training loss.

## Experiment Outputs

Experiment results are organized by model, experiment type, and run name:

```text
outputs/
  <model>/
    <experiment_type>/
      <run_name>/
        config.json
        metrics.json
        best_model.pth
        predicted_samples_<model>.png
```

Experiment types include `baseline`, `learning_rate`, `loss_weights`, and `experiments`.

Each training run saves its configuration, summary metrics, checkpoint (unless disabled), and prediction visualization (unless disabled).

Per-epoch metrics are tracked through W&B rather than a separate CSV file.

Model checkpoints (`*.pth`) and local W&B files are excluded from Git, while experiment configurations, metrics, and prediction figures remain trackable.

### Evaluating a Checkpoint

To evaluate a previously trained model:

```bash
python scripts/evaluate.py \
  --config config.json \
  --checkpoint path/to/best_model.pth \
  --output-dir outputs/manet/experiments/evaluation_001 \
  --no-show
```

Use the configuration associated with the checkpoint.

## Testing

Run the project verification script:

```bash
python -B scripts/check_project.py
```

Run the unit tests:

```bash
python -B -m unittest discover -s tests -v
```

The tests cover configuration handling, dataset processing, loss functions, IoU calculations, training metrics, checkpoint selection, and output management.