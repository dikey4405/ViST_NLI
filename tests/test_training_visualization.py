from __future__ import annotations

import csv
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from KLTN.source.training.visualization import (
    extract_dev_f1_points,
    extract_learning_rate_points,
    extract_loss_points,
    normalize_confusion_rows,
    plot_loss_curves,
    plot_training_dynamics,
    save_confusion_artifacts,
)


class TestTrainingVisualization(unittest.TestCase):
    def test_loss_points_keep_recorded_epochs_and_values(self) -> None:
        history = [
            {"epoch": 1, "train": {"total_loss": 1.2}, "dev": {"total_loss": 1.0}},
            {"epoch": 2, "train": {"total_loss": 0.9}, "dev": {}},
            {"epoch": 3, "train": {"total_loss": 0.7}, "dev": {"total_loss": 0.8}},
        ]
        self.assertEqual(
            extract_loss_points(history, split="train", metric="total_loss"),
            [(1, 1.2), (2, 0.9), (3, 0.7)],
        )
        self.assertEqual(
            extract_loss_points(history, split="dev", metric="total_loss"),
            [(1, 1.0), (3, 0.8)],
        )

    def test_plots_use_only_current_run_and_close_figures(self) -> None:
        history = [
            {
                "run_id": "first", "epoch": 1,
                "train": {"total_loss": 1.2, "classification_loss": 1.0},
                "dev": {"total_loss": 1.1, "classification_loss": 0.9},
            },
            {
                "run_id": "other", "epoch": 1,
                "train": {"total_loss": 999.0, "classification_loss": 999.0},
                "dev": {"total_loss": 999.0, "classification_loss": 999.0},
            },
            {
                "run_id": "first", "epoch": 2,
                "train": {"total_loss": 0.8, "classification_loss": 0.7},
                "dev": {"total_loss": 0.9, "classification_loss": 0.8},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            before = plt.get_fignums()
            with patch(
                "KLTN.source.training.visualization.extract_loss_points",
                wraps=extract_loss_points,
            ) as extract:
                paths = plot_loss_curves(
                    history,
                    output_dir=Path(directory),
                    run_id="first",
                    best_epoch=2,
                    logger=logging.getLogger(__name__),
                )
            self.assertEqual(plt.get_fignums(), before)
            self.assertEqual(len(paths), 2)
            self.assertEqual(
                {path.name for path in paths},
                {"loss_total_first.png", "loss_classification_first.png"},
            )
            for call in extract.call_args_list:
                self.assertEqual(
                    [record["run_id"] for record in call.args[0]],
                    ["first", "first"],
                )
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertGreaterEqual(image.info["dpi"][0], 299)
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    def test_missing_dev_is_warned_not_imputed(self) -> None:
        history = [
            {
                "run_id": "train-only", "epoch": 1, "learning_rate": 1e-4,
                "train": {"total_loss": 1.0, "classification_loss": 0.9},
                "dev": {},
            }
        ]
        logger = logging.getLogger(__name__)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(logger, "warning") as warning:
                paths = plot_loss_curves(
                    history,
                    output_dir=Path(directory),
                    run_id="train-only",
                    best_epoch=1,
                    logger=logger,
                )
        self.assertEqual(len(paths), 2)
        self.assertEqual(warning.call_count, 2)

    def test_dynamics_uses_recorded_epoch_lr_and_percent_f1(self) -> None:
        history = [
            {
                "run_id": "chosen", "epoch": epoch, "learning_rate": rate,
                "train": {"total_loss": train_loss, "classification_loss": train_loss / 2},
                "dev": {
                    "total_loss": dev_loss, "classification_loss": dev_loss / 2,
                    "macro_f1": f1,
                },
            }
            for epoch, rate, train_loss, dev_loss, f1 in (
                (1, 5e-5, 1.2, 1.0, 0.70),
                (2, 4e-5, 0.9, 1.1, 0.81),
                (3, 3e-5, 0.7, 1.3, 0.78),
            )
        ]
        history.insert(1, {
            "run_id": "other", "epoch": 1, "learning_rate": 0.5,
            "train": {"total_loss": 99, "classification_loss": 99},
            "dev": {"total_loss": 99, "classification_loss": 99, "macro_f1": 0.99},
        })
        self.assertEqual(
            extract_learning_rate_points(history[:1]), [(1, 5e-5)]
        )
        self.assertEqual(extract_dev_f1_points(history[:1]), [(1, 70.0)])

        figures = []
        original_subplots = plt.subplots

        def capture_subplots(*args, **kwargs):
            result = original_subplots(*args, **kwargs)
            figures.append(result[0])
            return result

        with tempfile.TemporaryDirectory() as directory:
            before = plt.get_fignums()
            with patch(
                "KLTN.source.training.visualization.plt.subplots",
                side_effect=capture_subplots,
            ):
                path = plot_training_dynamics(
                    history,
                    output_dir=Path(directory),
                    run_id="chosen",
                    best_epoch=2,
                    logger=logging.getLogger(__name__),
                )
            self.assertIsNotNone(path)
            self.assertEqual(path.name, "training_dynamics_chosen.png")
            with Image.open(path) as image:
                self.assertGreaterEqual(image.info["dpi"][0], 299)
            self.assertEqual(plt.get_fignums(), before)

        axes = figures[0].axes
        self.assertEqual(len(axes), 4)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [5e-5, 4e-5, 3e-5])
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [1.2, 0.9, 0.7])
        np.testing.assert_allclose(axes[1].lines[1].get_ydata(), [1.0, 1.1, 1.3])
        np.testing.assert_allclose(axes[2].lines[0].get_ydata(), [0.6, 0.45, 0.35])
        np.testing.assert_allclose(axes[3].lines[0].get_ydata(), [70, 81, 78])
        np.testing.assert_allclose(axes[3].collections[0].get_offsets(), [[2, 81]])
        for ax in axes:
            self.assertEqual(ax.lines[-1].get_xdata()[0], 2)

    def test_missing_epoch_lr_has_gap_and_no_scheduler_is_flat(self) -> None:
        history = [
            {"run_id": "old", "epoch": 1, "learning_rate": 1e-4,
             "train": {"total_loss": 1.0, "classification_loss": 0.8}, "dev": {}},
            {"run_id": "old", "epoch": 2,
             "train": {"total_loss": 0.9, "classification_loss": 0.7}, "dev": {}},
            {"run_id": "old", "epoch": 3, "learning_rate": 1e-4,
             "train": {"total_loss": 0.8, "classification_loss": 0.6}, "dev": {}},
        ]
        figures = []
        original_subplots = plt.subplots

        def capture_subplots(*args, **kwargs):
            result = original_subplots(*args, **kwargs)
            figures.append(result[0])
            return result

        with tempfile.TemporaryDirectory() as directory:
            with patch("KLTN.source.training.visualization.plt.subplots", side_effect=capture_subplots):
                paths = plot_loss_curves(
                    history, output_dir=Path(directory), run_id="old",
                    best_epoch=3, logger=logging.getLogger(__name__),
                )
            self.assertEqual(len(paths), 2)
        self.assertEqual(len(figures[0].axes), 2)
        lr_values = figures[0].axes[1].lines[0].get_ydata()
        self.assertEqual(lr_values[0], 1e-4)
        self.assertTrue(np.isnan(lr_values[1]))
        self.assertEqual(lr_values[2], 1e-4)

    def test_legacy_history_without_lr_omits_lr_panel(self) -> None:
        history = [
            {
                "run_id": "legacy", "epoch": 1,
                "train": {"total_loss": 1.0, "classification_loss": 0.8},
                "dev": {"total_loss": 1.1, "classification_loss": 0.9,
                        "macro_f1": 0.6},
            }
        ]
        figures = []
        original_subplots = plt.subplots

        def capture_subplots(*args, **kwargs):
            result = original_subplots(*args, **kwargs)
            figures.append(result[0])
            return result

        logger = logging.getLogger(__name__)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(logger, "warning") as warning, patch(
                "KLTN.source.training.visualization.plt.subplots",
                side_effect=capture_subplots,
            ):
                path = plot_training_dynamics(
                    history, output_dir=Path(directory), run_id="legacy",
                    best_epoch=1, logger=logger,
                )
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())
        self.assertEqual(len(figures[0].axes), 3)
        self.assertTrue(any("learning rate" in call.args[0] for call in warning.call_args_list))

    def test_confusion_csv_and_zero_support_row(self) -> None:
        confusion = torch.tensor([[2, 1, 0], [0, 0, 0], [1, 0, 3]])
        percentages = normalize_confusion_rows(confusion.numpy())
        np.testing.assert_allclose(percentages[0], [200 / 3, 100 / 3, 0])
        np.testing.assert_array_equal(percentages[1], [0, 0, 0])
        self.assertTrue(np.isfinite(percentages).all())

        with tempfile.TemporaryDirectory() as directory:
            before = plt.get_fignums()
            paths = save_confusion_artifacts(
                confusion,
                split="dev",
                output_dir=Path(directory),
                run_id="example",
            )
            self.assertEqual(plt.get_fignums(), before)
            self.assertEqual(len(paths), 3)
            self.assertEqual(
                {path.name for path in paths},
                {
                    "confusion_matrix_dev_example.csv",
                    "confusion_matrix_dev_example.png",
                    "confusion_matrix_dev_normalized_example.png",
                },
            )
            with (Path(directory) / "confusion_matrix_dev_example.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows, [
                ["true_label/predicted_label", "Entailment", "Neutral", "Contradiction"],
                ["Entailment", "2", "1", "0"],
                ["Neutral", "0", "0", "0"],
                ["Contradiction", "1", "0", "3"],
            ])
            for path in paths:
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        self.assertGreaterEqual(image.info["dpi"][0], 299)

    def test_invalid_confusion_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for value in (torch.zeros(2, 2, dtype=torch.long), torch.full((3, 3), -1)):
                with self.subTest(shape=tuple(value.shape)):
                    with self.assertRaises(ValueError):
                        save_confusion_artifacts(
                            value,
                            split="test",
                            output_dir=Path(directory),
                            run_id="invalid",
                        )


if __name__ == "__main__":
    unittest.main()
