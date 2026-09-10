from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.data_loader import GestureWindowDataset, build_train_loaders
from utils.mock_data import generate_mock_sequence
from utils.schema import BACKGROUND_ID, IGNORE_INDEX, save_sequence
from utils.split import load_manifest, split_video_files, write_manifest


def write_records(root: Path, count: int = 6) -> list[Path]:
    files: list[Path] = []
    for index in range(count):
        record = generate_mock_sequence(
            video_id=f"video_{index:02d}",
            num_frames=120,
            fps=15.0,
            seed=10 + index,
        )
        path = root / f"{record.video_id}.npz"
        save_sequence(path, record)
        files.append(path)
    return files


class SplitAndDatasetTests(unittest.TestCase):
    def test_video_level_split_has_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = write_records(root)
            splits = split_video_files(files, 0.5, 0.25, 0.25, seed=7)
            ids = {
                name: {path.stem for path in items}
                for name, items in splits.items()
            }
            self.assertTrue(ids["train"].isdisjoint(ids["validation"]))
            self.assertTrue(ids["train"].isdisjoint(ids["test"]))
            self.assertTrue(ids["validation"].isdisjoint(ids["test"]))
            manifest = write_manifest(
                root / "splits.json",
                splits,
                seed=7,
                ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
                processed_dir=root,
            )
            loaded = load_manifest(manifest)
            self.assertEqual(len(loaded["train"]), len(splits["train"]))

    def test_window_dataset_masks_and_background_sampling(self) -> None:
        record = generate_mock_sequence("window_demo", 80, 15.0, seed=3)
        record.labels[:] = BACKGROUND_ID
        record.labels[10:40] = 1
        dataset = GestureWindowDataset(
            [record],
            window_size=20,
            stride=20,
            keep_background_prob=0.0,
            seed=0,
        )
        starts = {sample.start for sample in dataset.windows}
        self.assertIn(0, starts)
        self.assertIn(20, starts)
        self.assertNotIn(60, starts)
        item = dataset[0]
        self.assertEqual(tuple(item["features"].shape), (128, 20))
        self.assertEqual(tuple(item["labels"].shape), (20,))
        self.assertTrue((item["valid_mask"] >= 0).all())

    def test_short_video_is_padded_with_ignore_index(self) -> None:
        record = generate_mock_sequence("short", 10, 15.0, seed=1)
        dataset = GestureWindowDataset(
            [record],
            window_size=16,
            stride=16,
            keep_background_prob=1.0,
            seed=0,
        )
        labels = dataset[0]["labels"].numpy()
        self.assertEqual(labels.shape[0], 16)
        self.assertTrue(np.all(labels[10:] == IGNORE_INDEX))

    def test_build_train_loaders_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = write_records(root, count=6)
            splits = split_video_files(files, 0.5, 0.25, 0.25, seed=2)
            manifest = write_manifest(
                root / "splits.json",
                splits,
                seed=2,
                ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
                processed_dir=root,
            )
            config = {
                "seed": 2,
                "data": {
                    "manifest_path": str(manifest),
                    "window_size": 30,
                    "stride": 15,
                    "keep_background_prob": 0.5,
                    "num_workers": 0,
                },
                "training": {"batch_size": 2},
            }
            train_loader, val_loader, train_files, val_files = build_train_loaders(config)
            batch = next(iter(train_loader))
            self.assertEqual(tuple(batch["features"].shape[1:]), (128, 30))
            self.assertGreater(len(train_files), 0)
            self.assertGreater(len(val_files), 0)
            self.assertGreater(len(val_loader.dataset), 0)


if __name__ == "__main__":
    unittest.main()
