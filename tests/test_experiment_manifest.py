from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.preprocessing.manifest import (
    ManifestConfig,
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


if __name__ == "__main__":
    unittest.main()
