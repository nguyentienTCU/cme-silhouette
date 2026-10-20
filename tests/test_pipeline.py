import contextlib
import csv
import io
import json
import importlib.util
from pathlib import Path
import shutil
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.configuration import PROJECT_ROOT, create_output_dir, load_config, output_path
from src.datasets import ProcessedSegmentationDataset, build_dataloaders
from src.losses import combined_loss
from src.metrics import iou_metric
from src.models import build_model
from src.trainer import fit, run_training, train_one_epoch
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
                self.assertEqual(set(metrics), {
                    'final', 'best_validation', 'best_val_iou', 'best_val_iou_epoch',
                    'checkpoint_selection', 'best_training', 'checkpoint_saved'
                })
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


class MockTrialPruned(Exception):
    """Stand-in for Optuna's exception when the optional library is unavailable."""


class PruningTests(unittest.TestCase):
    def test_trial_reports_each_epoch_without_changing_completed_metrics(self):
        config = dict(num_epochs=3, alpha_loss=0.7, checkpoint_path='best_model.pth')
        trial = Mock()
        trial.should_prune.return_value = False
        results = []
        for current_trial in (None, trial):
            with temporary_workspace() as directory:
                root = Path(directory)
                with patch('src.trainer.train_one_epoch', side_effect=[(3., .1), (2., .2), (2.5, .3)]), \
                     patch('src.trainer.validate', side_effect=[(1., .8), (2., .5), (.5, .9)]), \
                     patch('src.trainer.save_checkpoint') as save:
                    metrics = fit(None, None, None, None, 'cpu', config, root,
                                  save_checkpoints=True, trial=current_trial)
                results.append(metrics)
                self.assertEqual(save.call_count, 2)
                self.assertEqual([c.args[2] for c in save.call_args_list], [0, 1])
                self.assertEqual(json.loads((root / 'metrics.json').read_text()), metrics)
        self.assertEqual(results[0], results[1])
        self.assertEqual(trial.report.call_args_list,
                         [call(.8, step=0), call(.5, step=1), call(.9, step=2)])
        self.assertEqual(trial.should_prune.call_count, 3)
        self.assertEqual(results[1]['best_training']['epoch'], 2)
        self.assertEqual(results[1]['best_val_iou'], .9)

    def test_pruning_stops_early_after_logging_and_retains_configuration(self):
        config = load_config(PROJECT_ROOT / 'config.json')
        trial = Mock()
        wandb = SimpleNamespace(init=Mock(), finish=Mock(), log=Mock(), config=Mock())
        optuna_api = SimpleNamespace(TrialPruned=MockTrialPruned)
        model = torch.nn.Conv2d(1, 1, 1)
        loader = SimpleNamespace(dataset=[None] * 2)
        with temporary_workspace() as directory, \
             patch.dict('sys.modules', {'wandb': wandb, 'optuna': optuna_api}), \
             patch('src.trainer.build_dataloaders', return_value=(loader, loader)), \
             patch('src.trainer.build_model', return_value=model), \
             patch('src.trainer.torch.cuda.is_available', return_value=False), \
             patch('src.trainer.train_one_epoch', side_effect=[(3., .1), (2., .2)]) as train, \
             patch('src.trainer.validate', side_effect=[(1., .8), (2., .5)]) as validation, \
             patch('src.trainer.save_checkpoint') as save, \
             patch('src.evaluator.visualize_predictions') as visualize:
            def should_prune():
                # Ensure the current epoch's metrics have already reached W&B.
                self.assertEqual(wandb.log.call_count, trial.report.call_count)
                return trial.report.call_count == 2
            trial.should_prune.side_effect = should_prune
            with self.assertRaises(MockTrialPruned):
                run_training(config, output_root=directory, save_checkpoints=False,
                             visualize=False, show=False, trial=trial)
            path = output_path(config, directory)
            saved_config = json.loads((path / 'config.json').read_text())
            self.assertEqual(saved_config['run_name'], config['run_name'])
            self.assertFalse(saved_config['save_checkpoints'])
            self.assertFalse((path / 'metrics.json').exists())
            self.assertFalse((path / 'best_model.pth').exists())
            self.assertEqual(train.call_count, 2)
            self.assertEqual(validation.call_count, 2)
            self.assertEqual(trial.report.call_args_list, [call(.8, step=0), call(.5, step=1)])
            self.assertEqual(wandb.log.call_count, 2)
            wandb.finish.assert_called_once_with()
            save.assert_not_called()
            visualize.assert_not_called()

    def test_median_pruner_objective_and_total_attempt_count(self):
        import yaml
        config = yaml.safe_load((PROJECT_ROOT / 'configs/optuna/manet.yaml').read_text())
        self.assertEqual(config['pruning'], {
            'n_startup_trials': 3, 'n_warmup_steps': 5, 'interval_steps': 1
        })
        self.assertEqual(config['search_space']['batch_size'], {'low': 30, 'high': 60, 'step': 1})
        for recorded_count in (3, 10, 12):
            with self.subTest(recorded_count=recorded_count), temporary_workspace() as directory:
                study = Mock(best_value=.8, best_params={'learning_rate': 1e-5})
                # Mixed states are all recorded attempts, not only successful trials.
                study.trials = [SimpleNamespace(state=state) for state in
                                (['COMPLETE', 'FAIL', 'PRUNED'] * 4)[:recorded_count]]
                optuna_api = SimpleNamespace(pruners=SimpleNamespace(MedianPruner=Mock()),
                                              create_study=Mock(return_value=study))
                spec = importlib.util.spec_from_file_location(
                    'src.hyperparameter_optimization', PROJECT_ROOT / 'src/hyperparameter_optimization.py')
                module = importlib.util.module_from_spec(spec)
                with patch.dict('sys.modules', {'optuna': optuna_api}):
                    spec.loader.exec_module(module)
                trials = []
                scores = []
                def optimize(objective, n_trials):
                    for number in range(recorded_count, recorded_count + n_trials):
                        trial = Mock(number=number)
                        trial.suggest_float.side_effect = [1e-5, .7]
                        trial.suggest_int.return_value = 30
                        trials.append(trial)
                        scores.append(objective(trial))
                study.optimize.side_effect = optimize
                base = load_config(PROJECT_ROOT / 'config.json')
                with patch.object(module, 'PROJECT_ROOT', Path(directory)), \
                     patch.object(module, 'load_config', return_value=base), \
                     patch.object(module, 'run_training', return_value={
                         'metrics': {'best_val_iou': .8, 'final': {'val_iou': .2}},
                         'output_dir': 'trial-output'
                     }) as training, contextlib.redirect_stdout(io.StringIO()):
                    result = module.run_optimization(config)
                self.assertIs(result, study)
                optuna_api.pruners.MedianPruner.assert_called_once_with(**config['pruning'])
                storage_path = (Path(directory) / 'outputs/manet/optuna/study.db').resolve()
                optuna_api.create_study.assert_called_once_with(
                    study_name=config['study_name'], direction='maximize',
                    storage=f'sqlite:///{storage_path.as_posix()}', load_if_exists=True,
                    pruner=optuna_api.pruners.MedianPruner.return_value)
                remaining = max(0, config['n_trials'] - recorded_count)
                if remaining:
                    self.assertEqual(study.optimize.call_args.kwargs['n_trials'], remaining)
                else:
                    study.optimize.assert_not_called()
                self.assertEqual(scores, [.8] * remaining)
                names = []
                for training_call, trial in zip(training.call_args_list, trials):
                    trial_config = training_call.args[0]
                    names.append(trial_config['run_name'])
                    self.assertEqual(trial_config['run_name'], f"{config['study_name']}_trial_{trial.number}")
                    self.assertEqual(trial_config['experiment_type'], 'experiments')
                    self.assertEqual(training_call.kwargs, {
                        'output_dir': (storage_path.parent / 'trials' / trial_config['run_name']),
                        'save_checkpoints': False, 'visualize': False, 'show': False, 'trial': trial
                    })
                    trial.suggest_int.assert_called_once_with('batch_size', 30, 60, step=1)
                self.assertEqual(len(names), len(set(names)))

    @unittest.skipUnless(importlib.util.find_spec('optuna') is not None,
                         'Optuna unavailable locally; run this SQLite integration test on Nova')
    def test_sqlite_records_pruned_intermediate_values_and_resumes(self):
        import optuna
        with temporary_workspace() as directory:
            root = Path(directory)
            storage = f"sqlite:///{(root / 'study.db').as_posix()}"
            original_storage = optuna.storages.RDBStorage(storage)
            study = optuna.create_study(study_name='pruning_test', direction='maximize',
                                        storage=original_storage, load_if_exists=True,
                                        pruner=optuna.pruners.MedianPruner(
                                            n_startup_trials=1, n_warmup_steps=0, interval_steps=1))
            config = dict(num_epochs=3, alpha_loss=.7, checkpoint_path='best_model.pth')
            def objective(trial):
                trial_dir = root / f'trial_{trial.number}'
                trial_dir.mkdir()
                ious = [.8, .9, .7] if trial.number == 0 else [.1, .2, .3]
                with patch('src.trainer.train_one_epoch', return_value=(1., .2)), \
                     patch('src.trainer.validate', side_effect=[(1., value) for value in ious]):
                    metrics = fit(None, None, None, None, 'cpu', config, trial_dir,
                                  save_checkpoints=False, trial=trial)
                return metrics['best_val_iou']
            study.optimize(objective, n_trials=2)
            resumed_storage = optuna.storages.RDBStorage(storage)
            resumed = optuna.load_study(study_name='pruning_test', storage=resumed_storage)
            self.assertEqual(len(resumed.trials), 2)
            self.assertEqual(resumed.trials[0].state, optuna.trial.TrialState.COMPLETE)
            self.assertEqual(resumed.trials[0].value, .9)
            self.assertEqual(resumed.trials[1].state, optuna.trial.TrialState.PRUNED)
            self.assertEqual(resumed.trials[1].intermediate_values, {0: .1})
            self.assertFalse((root / 'trial_1/metrics.json').exists())
            # Release SQLite connections before removing the temporary directory on Windows.
            for backend in (original_storage, resumed_storage):
                backend.remove_session()
                backend.engine.dispose()


if __name__ == '__main__':
    unittest.main()
