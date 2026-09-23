from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.preprocessing.manifest import (
    ManifestConfig,
    _load_dirty_video_ids,
    build_experiment_manifest,
)


class ExperimentManifestTests(unittest.TestCase):
    def test_subject_split_is_global_across_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            audit = root / "audit.csv"
            rows = []
            for subject_index in range(8):
                for source in ("data_lm", "data_sr"):
                    path = processed / source / f"{source}_{subject_index}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    rows.append(
                        {
                            "video_id": path.stem,
                            "source": source,
                            "subject_id": f"subject_{subject_index}",
                            "scene": "stand",
                            "scene_raw": "stand",
                            "polarity": "positive",
                            "scene_category": "posture",
                            "hard_negative_eligible": "False",
                            "quality_passed": "True",
                            "sequence_path": str(path),
                            "preprocess_fingerprint": "recipe",
                        }
                    )
            with audit.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = ManifestConfig(
                name="all5",
                output_dir=str(root / "manifests"),
                processed_dir=str(processed),
                audit_path=str(audit),
                subject_splits_path=str(root / "subject_splits.json"),
                seed=7,
            )
            output = build_experiment_manifest(config)
            payload = json.loads(output.read_text(encoding="utf-8"))
            subject_locations: dict[str, set[str]] = {}
            for record in payload["records"]:
                subject_locations.setdefault(record["subject_id"], set()).add(
                    record["split"]
                )
            self.assertTrue(
                all(len(locations) == 1 for locations in subject_locations.values())
            )
            self.assertTrue(
                any("/" in item for item in payload["splits"]["train"])
            )

    def test_dirty_records_are_excluded_from_every_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed = root / "processed"
            audit = root / "audit.csv"
            rows = []
            for subject_index in range(8):
                for source in ("data_lm", "data_sr"):
                    video_id = f"{source}__sample_{subject_index}"
                    path = processed / source / f"{video_id}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                    rows.append(
                        {
                            "video_id": video_id,
                            "source": source,
                            "subject_id": f"subject_{subject_index}",
                            "scene": "stand",
                            "scene_raw": "stand",
                            "polarity": "positive",
                            "scene_category": "posture",
                            "hard_negative_eligible": "False",
                            "quality_passed": "True",
                            "sequence_path": str(path),
                            "preprocess_fingerprint": "recipe",
                        }
                    )
            with audit.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            dirty_data = root / "dirty_data.txt"
            dirty_data.write_text(
                "# manually rejected recordings\n"
                "data_lm\\sample_0\n"
                "data_sr/sample_1\n",
                encoding="utf-8",
            )
            config = ManifestConfig(
                name="filtered",
                output_dir=str(root / "manifests"),
                processed_dir=str(processed),
                audit_path=str(audit),
                dirty_data_path=str(dirty_data),
                subject_splits_path=str(root / "subject_splits.json"),
                seed=7,
            )

            output = build_experiment_manifest(config)

            payload = json.loads(output.read_text(encoding="utf-8"))
            dirty_ids = {"data_lm__sample_0", "data_sr__sample_1"}
            manifest_ids = {record["video_id"] for record in payload["records"]}
            split_paths = {
                item
                for paths in payload["splits"].values()
                for item in paths
            }
            self.assertTrue(dirty_ids.isdisjoint(manifest_ids))
            self.assertTrue(
                all(not any(video_id in path for video_id in dirty_ids) for path in split_paths)
            )
            self.assertEqual(payload["dirty_data"]["listed_video_count"], 2)
            self.assertEqual(payload["dirty_data"]["excluded_record_count"], 2)
            self.assertEqual(
                payload["dirty_data"]["excluded_video_ids"],
                sorted(dirty_ids),
            )

    def test_dirty_data_list_fails_fast_when_missing_or_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                _load_dirty_video_ids(str(root / "missing.txt"))

            invalid = root / "dirty_data.txt"
            invalid.write_text(
                "missing-source-separator\nsource//sample\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, r"lines 1, 2"):
                _load_dirty_video_ids(str(invalid))


if __name__ == "__main__":
    unittest.main()
