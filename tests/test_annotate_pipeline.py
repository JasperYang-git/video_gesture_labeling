from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from utils.preprocessing.annotations import (
    intervals_from_prediction,
    labels_from_intervals,
    parse_nova_annotation,
    write_nova_annotation,
)
from annotate_videos import (
    annotation_output_path,
    forget_self_annotations,
    is_self_written,
    write_prediction_marker,
)
from utils.preprocessing.assemble import AssembleConfig, assemble_entry
from utils.preprocessing.inventory import InventoryEntry, scan_video_root
from utils.preprocessing.tracks import TrackConfig, save_track, track_cache_path
from utils.schema import BACKGROUND_ID, load_sequence
from utils.subtitles import format_timestamp, segments_to_srt


def make_prediction() -> np.ndarray:
    prediction = np.full(60, BACKGROUND_ID, dtype=np.int64)
    prediction[10:20] = 3
    prediction[40:46] = 9
    return prediction


class AnnotationWriterTests(unittest.TestCase):
    def test_write_then_parse_restores_the_same_frames(self) -> None:
        prediction = make_prediction()
        fps = 15.0
        intervals = intervals_from_prediction(prediction, fps)
        self.assertEqual([interval.label for interval in intervals], [3, 9])
        self.assertAlmostEqual(intervals[0].start_sec, 10 / fps)
        self.assertAlmostEqual(intervals[0].end_sec, 20 / fps)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "NOVA project" / "gestures.annotation~"
            write_nova_annotation(path, intervals)
            restored = labels_from_intervals(
                len(prediction), fps, parse_nova_annotation(path)
            )
        np.testing.assert_array_equal(restored, prediction)

    def test_round_trip_survives_timestamp_rounding(self) -> None:
        # Four-decimal timestamps plus the floor/ceil in labels_from_intervals used to
        # grow every segment by a frame, so sweep frame rates and boundaries.
        generator = np.random.default_rng(0)
        for fps in (15.0, 25.0, 29.97, 30.0, 60.0):
            for _ in range(20):
                prediction = np.full(120, BACKGROUND_ID, dtype=np.int64)
                start = int(generator.integers(0, 90))
                length = int(generator.integers(1, 25))
                prediction[start : start + length] = int(generator.integers(0, 14))
                with tempfile.TemporaryDirectory() as temp_dir:
                    path = Path(temp_dir) / "gestures.annotation~"
                    write_nova_annotation(
                        path, intervals_from_prediction(prediction, fps)
                    )
                    restored = labels_from_intervals(
                        len(prediction), fps, parse_nova_annotation(path)
                    )
                np.testing.assert_array_equal(restored, prediction)

    def test_confidence_comes_from_logits(self) -> None:
        prediction = np.array([2, 2, 2], dtype=np.int64)
        logits = np.zeros((15, 3), dtype=np.float32)
        logits[2, :] = 10.0
        intervals = intervals_from_prediction(prediction, 15.0, logits)
        self.assertEqual(len(intervals), 1)
        self.assertGreater(intervals[0].confidence, 0.99)

        flat = np.zeros((15, 3), dtype=np.float32)
        uncertain = intervals_from_prediction(prediction, 15.0, flat)
        self.assertAlmostEqual(uncertain[0].confidence, 1 / 15, places=5)

    def test_empty_prediction_writes_empty_file(self) -> None:
        prediction = np.full(10, BACKGROUND_ID, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gestures.annotation~"
            write_nova_annotation(path, intervals_from_prediction(prediction, 15.0))
            self.assertEqual(path.read_text(encoding="utf-8"), "")
            self.assertEqual(parse_nova_annotation(path), [])


class SubtitleTests(unittest.TestCase):
    def test_timestamp_format(self) -> None:
        self.assertEqual(format_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_timestamp(1.5), "00:00:01,500")
        self.assertEqual(format_timestamp(3723.456), "01:02:03,456")
        self.assertEqual(format_timestamp(-1.0), "00:00:00,000")

    def test_srt_blocks_are_numbered_and_named(self) -> None:
        intervals = intervals_from_prediction(make_prediction(), 15.0)
        srt = segments_to_srt(intervals)
        lines = srt.strip().splitlines()
        self.assertEqual(lines[0], "1")
        self.assertIn("-->", lines[1])
        self.assertIn("slide-down", lines[2])
        self.assertEqual(lines[4], "2")
        self.assertIn("double-knock", lines[6])


class ScanVideoRootTests(unittest.TestCase):
    # Real unlabeled recordings arrive as a flat dump with a naming scheme that has
    # nothing in common with the training folders, so both layouts must work.
    FLAT_NAMES = (
        "DJI_20260613115350_LN043-bike1-5.mp4",
        "VID_20260613_090940_LN044-stand4-5.mp4",
        "VID20260613154147_LN046-stand5-8.mp4",
    )

    def test_flat_directory_yields_one_sample_per_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in self.FLAT_NAMES:
                (root / name).touch()
            entries = scan_video_root(root)

        self.assertEqual(len(entries), len(self.FLAT_NAMES))
        self.assertEqual(len({entry.video_id for entry in entries}), len(entries))
        for entry in entries:
            stem = Path(entry.video_path).stem
            # Each loose file gets its own sample folder named after the file.
            self.assertEqual(Path(entry.directory_path), root / stem)
            self.assertEqual(entry.video_id, f"inference__{stem}")
            self.assertEqual(entry.status, "ok")
            self.assertIsNone(entry.annotation_path)
            # A name that does not follow the training convention must not be
            # mined for bogus metadata.
            self.assertEqual(entry.subject_id, "unknown")
            self.assertEqual(entry.gender, "unknown")
            self.assertEqual(entry.field3, "unknown")

    def test_training_layout_keeps_folder_and_parses_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labeled = root / "DG2024062702_F_R_sit"
            (labeled / "NOVA project").mkdir(parents=True)
            (labeled / "recording.mp4").touch()
            (labeled / "NOVA project" / "gestures.annotation~").write_text(
                "0;1;0;1;\n", encoding="utf-8"
            )
            entries = scan_video_root(root)

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(Path(entry.directory_path), labeled)
        self.assertEqual(entry.video_id, "inference__DG2024062702_F_R_sit")
        self.assertIsNotNone(entry.annotation_path)
        self.assertEqual(entry.subject_id, "DG2024062702")
        self.assertEqual(entry.field3, "R")

    def test_mixed_layouts_coexist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labeled = root / "DG2024062702_F_R_sit"
            (labeled / "NOVA project").mkdir(parents=True)
            (labeled / "recording.mp4").touch()
            (labeled / "NOVA project" / "gestures.annotation~").touch()
            batch = root / "batch1"
            batch.mkdir()
            for name in self.FLAT_NAMES:
                (batch / name).touch()
            (root / "empty").mkdir()
            entries = scan_video_root(root)

        self.assertEqual(len(entries), 1 + len(self.FLAT_NAMES))
        by_id = {entry.video_id: entry for entry in entries}
        self.assertIn("inference__DG2024062702_F_R_sit", by_id)
        nested = by_id["inference__batch1__DJI_20260613115350_LN043-bike1-5"]
        self.assertEqual(
            Path(nested.directory_path),
            batch / "DJI_20260613115350_LN043-bike1-5",
        )

    def test_muxed_output_is_not_rescanned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "clip_a.mp4").touch()
            (root / "clip_b.mp4").touch()
            # Simulate a previous --mux run so rescanning stays idempotent.
            (root / "clip_a").mkdir()
            (root / "clip_a" / "clip_a_pred.mp4").touch()
            entries = scan_video_root(root)

        self.assertEqual(
            sorted(Path(entry.video_path).name for entry in entries),
            ["clip_a.mp4", "clip_b.mp4"],
        )

    def test_missing_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                scan_video_root(Path(temp_dir) / "absent")


class AnnotationTargetTests(unittest.TestCase):
    def make_entry(self, directory: Path, annotation: Path | None):
        return InventoryEntry(
            source="inference",
            video_id="inference__clip",
            directory_path=str(directory),
            video_path=str(directory / "clip.mp4"),
            annotation_path=str(annotation) if annotation else None,
            subject_id="unknown",
            gender="unknown",
            field3="unknown",
            scene_raw="unknown",
            scene="unknown",
            polarity="unknown",
            status="ok",
        )

    def test_hand_annotation_is_never_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "sample"
            nova = directory / "NOVA project"
            nova.mkdir(parents=True)
            hand = nova / "gestures.annotation~"
            hand.write_text("1;2;3;1;\n", encoding="utf-8")
            entry = self.make_entry(directory, hand)

            path, shadowed = annotation_output_path(entry, overwrite=False)
            self.assertTrue(shadowed)
            self.assertEqual(path.name, "gestures.pred.annotation~")
            self.assertEqual(hand.read_text(encoding="utf-8"), "1;2;3;1;\n")

            forced, shadowed = annotation_output_path(entry, overwrite=True)
            self.assertFalse(shadowed)
            self.assertEqual(forced, hand)

    def test_own_prediction_is_overwritten_and_not_scored_as_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "sample"
            nova = directory / "NOVA project"
            nova.mkdir(parents=True)
            annotation = nova / "gestures.annotation~"
            annotation.write_text("1;2;3;1;\n", encoding="utf-8")
            entry = self.make_entry(directory, annotation)
            write_prediction_marker(annotation, entry, "model.pth")

            # Rewritten in place rather than piling up .pred. copies.
            path, shadowed = annotation_output_path(entry, overwrite=False)
            self.assertFalse(shadowed)
            self.assertEqual(path, annotation)
            # And it must not come back as ground truth on the next run.
            self.assertTrue(is_self_written(entry))
            self.assertIsNone(forget_self_annotations([entry])[0].annotation_path)

    def test_hand_annotation_without_marker_stays_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir) / "sample"
            nova = directory / "NOVA project"
            nova.mkdir(parents=True)
            annotation = nova / "gestures.annotation~"
            annotation.write_text("1;2;3;1;\n", encoding="utf-8")
            entry = self.make_entry(directory, annotation)

            self.assertFalse(is_self_written(entry))
            self.assertEqual(
                forget_self_annotations([entry])[0].annotation_path, str(annotation)
            )


class UnlabeledAssembleTests(unittest.TestCase):
    def make_entry(self, root: Path, annotation_path: Path | None):
        video = root / "video.mp4"
        video.write_bytes(b"video")
        return SimpleNamespace(
            video_id="inference__sample",
            source="inference",
            subject_id="unknown",
            gender="unknown",
            field3="unknown",
            scene="unknown",
            scene_raw="unknown",
            polarity="unknown",
            scene_category="unknown",
            hard_negative_eligible=False,
            taxonomy_version=1,
            video_path=video,
            annotation_path=annotation_path,
            status="ok",
        )

    def save_fake_track(self, entry, config: TrackConfig) -> None:
        frames = 30
        landmarks = np.zeros((frames, 21, 3), dtype=np.float32)
        landmarks[:, 9, 1] = 0.2
        save_track(
            track_cache_path(config, entry.source, entry.video_id),
            landmarks,
            np.ones(frames, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
            30.0,
            {
                "track_fingerprint": "track",
                "track_config_fingerprint": config.fingerprint(),
            },
        )

    def test_assemble_without_annotation_yields_no_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entry = self.make_entry(root, None)
            track_config = TrackConfig(cache_dir=str(root / "tracks"))
            self.save_fake_track(entry, track_config)
            assemble_config = AssembleConfig(output_dir=str(root / "processed"))
            result = assemble_entry(entry, track_config, assemble_config)

            self.assertEqual(result["status"], "completed")
            record = load_sequence(result["sequence_path"])
            self.assertIsNone(record.labels)
            self.assertEqual(record.features.shape, (128, 15))
            # The checkpoint compatibility check only reads this key, so an
            # unlabeled sequence stays usable with a model trained on labeled data.
            self.assertEqual(
                record.metadata["preprocess_config_fingerprint"],
                assemble_config.fingerprint(),
            )

    def test_annotation_when_present_still_produces_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotation = root / "gestures.annotation~"
            annotation.write_text("0;0.5;0;1;\n", encoding="utf-8")
            entry = self.make_entry(root, annotation)
            track_config = TrackConfig(cache_dir=str(root / "tracks"))
            self.save_fake_track(entry, track_config)
            assemble_config = AssembleConfig(output_dir=str(root / "processed"))
            result = assemble_entry(entry, track_config, assemble_config)

            record = load_sequence(result["sequence_path"])
            self.assertIsNotNone(record.labels)
            self.assertEqual(len(record.labels), record.features.shape[1])
            self.assertEqual(
                record.metadata["preprocess_config_fingerprint"],
                assemble_config.fingerprint(),
            )


if __name__ == "__main__":
    unittest.main()
