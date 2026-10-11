import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.configuration import PROJECT_ROOT, create_output_dir, load_config, output_path
from src.datasets import ProcessedSegmentationDataset, build_dataloaders
from src.losses import combined_loss
from src.metrics import iou_metric
from src.models import build_model
from src.trainer import fit, train_one_epoch
from src.evaluator import validate


@contextlib.contextmanager
def temporary_workspace():
    # Normal inherited permissions work for Windows sandbox identities; Python's
    # private mode-0700 temporary folders exclude those identities on this host.
    directory = (PROJECT_ROOT / ('.test-tmp-' + uuid.uuid4().hex)).resolve()
    directory.mkdir()
    try:
        yield str(directory)
    finally:
        if directory.parent != PROJECT_ROOT.resolve() or not directory.name.startswith('.test-tmp-'):
            raise RuntimeError('Refusing to clean a directory outside the test workspace')
        shutil.rmtree(directory)


class PipelineTests(unittest.TestCase):
    def test_configuration_migration_and_entry_points(self):
        config = load_config(PROJECT_ROOT / 'config.json')
        self.assertEqual(config, load_config(PROJECT_ROOT / 'configs/models/manet.yaml'))
        for name in ('fpn', 'linknet', 'upernet'):
            historical = load_config(PROJECT_ROOT / f'configs/models/{name}.yaml')
            self.assertEqual(historical['model_name'], name)
            self.assertEqual(historical['learning_rate'], 0.0001)
        with self.assertRaisesRegex(ValueError, 'Missing configuration values'):
            load_config(PROJECT_ROOT / 'configs/models/deeplabv3plus.yaml')
        from scripts.train import main
        import train_model
        from scripts.evaluate import main as evaluate_main
        self.assertIs(train_model.main, main)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            main(['--dry-run'])
        self.assertEqual(json.loads(output.getvalue())['config'], config)
        self.assertTrue(callable(evaluate_main))

    def test_output_classification_and_collision(self):
        config = load_config(PROJECT_ROOT / 'config.json')
        with temporary_workspace() as directory:
            path = create_output_dir(config, directory)
            self.assertEqual(path, Path(directory) / 'manet/loss_weights' / config['run_name'])
            with self.assertRaises(FileExistsError):
                create_output_dir(config, directory)
            for name, category in [('manet_lr_1e-5', 'learning_rate'), ('manet_baseline', 'baseline'), ('unknown', 'experiments')]:
                candidate = dict(config, run_name=name)
                self.assertEqual(output_path(candidate, directory).parent.name, category)
            with self.assertRaises(ValueError):
                output_path(dict(config, run_name='../outside'), directory)

    def test_dataset_and_split(self):
        with temporary_workspace() as directory:
            root = Path(directory)
            np.save(root / 'sample.npy', {'input': np.array([[np.nan, 3.0], [1.0, 2.0]]),
                                        'mask': np.array([[[0], [255]], [[1], [0]]])})
            with (root / 'files.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['file_path'])
                writer.writerows([['sample.npy']] * 10)
            dataset = ProcessedSegmentationDataset(root / 'files.csv', root, (4, 4))
            image, mask = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 4, 4))
            self.assertEqual(image[0, 1, 1].item(), 0)
            self.assertEqual(mask.sum().item(), 2)
            config = dict(csv_file=str(root / 'files.csv'), root_dir=str(root), target_size=[4, 4],
                          seed=42, batch_size=3, num_workers=0)
            train, val = build_dataloaders(config)
            self.assertEqual((len(train.dataset), len(val.dataset)), (8, 2))
            self.assertTrue(train.drop_last)
            self.assertFalse(val.drop_last)
            train2, val2 = build_dataloaders(config)
            self.assertEqual(val.dataset.indices, val2.dataset.indices)

    def test_original_loss_and_iou_definitions(self):
        logits = torch.tensor([[[[-2., 0.], [2., 1.]]], [[[0., 3.], [-3., 1.]]]], requires_grad=True)
        targets = torch.tensor([[[[0., 1.], [1., 0.]]], [[[1., 1.], [0., 0.]]]])
        probabilities = logits.sigmoid()
        axes = (1, 2, 3)
        dice = (1 - (2 * (probabilities * targets).sum(axes) + 1e-6) /
                (probabilities.sum(axes) + targets.sum(axes) + 1e-6)).mean()
        for alpha in (0.3, 0.5, 0.7):
            expected = alpha * torch.nn.functional.binary_cross_entropy_with_logits(logits, targets) + (1-alpha) * dice
            actual = combined_loss(logits, targets, alpha=alpha)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                       torch.autograd.grad(expected, logits, retain_graph=True)[0])
        preds = (probabilities > 0.5).float()
        intersection = (preds * targets).sum(axes)
        expected_iou = ((intersection + 1e-6) / (preds.sum(axes) + targets.sum(axes) - intersection + 1e-6)).mean().item()
        self.assertEqual(iou_metric(logits, targets), expected_iou)

    def test_cpu_epoch_and_validation_aggregation(self):
        model = torch.nn.Conv2d(1, 1, 1)
        inputs = torch.arange(12, dtype=torch.float32).reshape(3, 1, 2, 2) / 12
        targets = (inputs > 0.5).float()
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        before = model.weight.detach().clone()
        loss, iou = train_one_epoch(model, loader, optimizer, 'cpu', 0.7)
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(iou, 0)
        self.assertFalse(torch.equal(before, model.weight))
        expected_loss = expected_iou = 0
        with torch.no_grad():
            for x, y in loader:
                out = model(x)
                expected_loss += combined_loss(out, y, alpha=0.7).item() * len(x)
                expected_iou += iou_metric(out, y) * len(x)
        actual_loss, actual_iou = validate(model, loader, 'cpu', 0.7)
        self.assertEqual(actual_loss, expected_loss / 3)
        self.assertEqual(actual_iou, expected_iou / 3)

    def test_checkpoint_criterion_and_lightweight_storage(self):
        model = torch.nn.Conv2d(1, 1, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        config = dict(
            num_epochs=3,
            alpha_loss=0.7,
            checkpoint_path='best_model.pth'
        )

        for save in (True, False):
            with temporary_workspace() as directory:
                root = Path(directory)

                with patch(
                    'src.trainer.train_one_epoch',
                    side_effect=[(3., .1), (2., .2), (2.5, .3)]
                ), patch(
                    'src.trainer.validate',
                    side_effect=[(1., .8), (2., .5), (.5, .9)]
                ):
                    metrics = fit(
                        model, None, None, optimizer, 'cpu',
                        config, root, save_checkpoints=save
                    )

                self.assertEqual(metrics['best_training']['epoch'], 2)
                self.assertEqual(metrics['best_validation']['epoch'], 3)
                self.assertEqual(metrics['final']['epoch'], 3)
                self.assertEqual(metrics['best_val_iou'], 0.9)
                self.assertEqual(metrics['best_val_iou_epoch'], 3)

                self.assertEqual(
                    (root / 'best_model.pth').exists(), save
                )

                metrics_path = root / 'metrics.json'
                self.assertTrue(metrics_path.exists())

                with metrics_path.open(encoding='utf-8') as stream:
                    saved_metrics = json.load(stream)

                self.assertEqual(saved_metrics, metrics)
                self.assertFalse((root / 'history.csv').exists())

                if save:
                    checkpoint = torch.load(
                        root / 'best_model.pth', weights_only=True
                    )
                    self.assertEqual(checkpoint['epoch'], 1)
                    self.assertEqual(checkpoint['loss'], 2.)
                    self.assertEqual(
                        set(checkpoint),
                        {
                            'epoch',
                            'model_state_dict',
                            'optimizer_state_dict',
                            'loss'
                        }
                    )

    def test_model_initialization_arguments(self):
        from unittest.mock import Mock
        constructor = Mock()
        build_model(constructor)
        constructor.assert_called_once_with(encoder_name='resnet18', encoder_weights='imagenet',
                                            in_channels=1, classes=1, activation=None)


if __name__ == '__main__':
    unittest.main()
