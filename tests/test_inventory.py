from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.preprocessing.inventory import (
    load_taxonomy,
    normalize_scene,
    read_inventory_csv,
    read_inventory_jsonl,
    resolve_scene,
    scan_inventory,
    write_inventory_csv,
    write_inventory_jsonl,
    write_summary_json,
)


TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "config" / "scene_taxonomy.yaml"


def _make_recording(root: Path, source: str, relative: str) -> Path:
    folder = root / source / relative
    (folder / "NOVA project").mkdir(parents=True)
    (folder / "recording.MP4").touch()
    (folder / "NOVA project" / "gestures.annotation~").touch()
    return folder


class InventoryTests(unittest.TestCase):
    def test_parses_name_without_truncating_subject_or_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            folder = _make_recording(
                root,
                "phone",
                "batch/DG20260609152055-LN023_F_meta_gesture-sit-naozhong_extra",
            )

            entries = scan_inventory(root, ["phone"], TAXONOMY_PATH)

            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.subject_id, "DG20260609152055-LN023")
            self.assertEqual(entry.gender, "F")
            self.assertEqual(entry.field3, "meta")
            self.assertEqual(entry.scene_raw, "gesture-sit-naozhong")
            self.assertEqual(entry.scene, "gesture-sit-naozhong")
            self.assertEqual(entry.polarity, "positive")
            self.assertEqual(
                entry.video_id,
                "phone__batch__DG20260609152055-LN023_F_meta_gesture-sit-naozhong_extra",
            )
            self.assertEqual(entry.video_path, str(folder / "recording.MP4"))
            self.assertEqual(entry.status, "ok")

    def test_unknown_has_no_polarity_and_no_prefix_guessing(self) -> None:
        taxonomy = load_taxonomy(TAXONOMY_PATH)
        self.assertEqual(normalize_scene("  LIE_UP  "), "lie-up")
        self.assertEqual(resolve_scene("stand-alarm-strong", taxonomy), (None, None))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_recording(root, "watch", "S1_M_meta_stand-alarm-strong")
            entry = scan_inventory(root, ["watch"], TAXONOMY_PATH)[0]
            self.assertIsNone(entry.scene)
            self.assertIsNone(entry.polarity)
            self.assertEqual(entry.status, "ok")

    def test_same_relative_name_in_two_sources_has_distinct_video_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            relative = "batch/S1_F_meta_sit"
            _make_recording(root, "source-a", relative)
            _make_recording(root, "source-b", relative)

            entries = scan_inventory(root, ["source-a", "source-b"], TAXONOMY_PATH)

            self.assertEqual(
                {entry.video_id for entry in entries},
                {
                    "source-a__batch__S1_F_meta_sit",
                    "source-b__batch__S1_F_meta_sit",
                },
            )

    def test_missing_files_and_invalid_name_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            no_video = root / "source" / "S1_F_meta_work"
            (no_video / "NOVA project").mkdir(parents=True)
            (no_video / "NOVA project" / "gestures.annotation~").touch()

            no_annotation = root / "source" / "S2_M_meta_walk"
            no_annotation.mkdir(parents=True)
            (no_annotation / "video.mp4").touch()

            invalid = root / "source" / "bad-name"
            (invalid / "NOVA project").mkdir(parents=True)
            (invalid / "video.mp4").touch()
            (invalid / "NOVA project" / "gestures.annotation~").touch()

            entries = scan_inventory(root, ["source"], TAXONOMY_PATH)
            by_name = {Path(entry.directory_path).name: entry for entry in entries}

            self.assertEqual(len(entries), 3)
            self.assertIn("missing_mp4", by_name["S1_F_meta_work"].error or "")
            self.assertEqual(by_name["S1_F_meta_work"].polarity, "negative")
            self.assertIn(
                "missing_nova_annotation", by_name["S2_M_meta_walk"].error or ""
            )
            self.assertIn("invalid_name", by_name["bad-name"].error or "")
            self.assertTrue(all(entry.status == "error" for entry in entries))

    def test_jsonl_csv_and_summary_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _make_recording(root, "source", "S1_F_meta_sit")
            entries = scan_inventory(root, ["source"], TAXONOMY_PATH)
            jsonl_path = root / "out" / "inventory.jsonl"
            csv_path = root / "out" / "inventory.csv"
            summary_path = root / "out" / "summary.json"

            write_inventory_jsonl(jsonl_path, entries)
            write_inventory_csv(csv_path, entries)
            write_summary_json(summary_path, entries)

            self.assertEqual(read_inventory_jsonl(jsonl_path), entries)
            self.assertEqual(read_inventory_csv(csv_path), entries)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["total"], 1)
            self.assertEqual(summary["by_status"], {"ok": 1})


if __name__ == "__main__":
    unittest.main()
