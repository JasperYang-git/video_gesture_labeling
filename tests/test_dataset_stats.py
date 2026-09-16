from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.data_loader import _action_segments
from utils.preprocessing.stats import (
    SourceGroup,
    action_segment_class_ids,
    build_groups,
    collect_group_stats,
    discover_sources,
    render_group_stats,
    write_stats_outputs,
)
from utils.schema import (
    BACKGROUND_ID,
    CLASS_NAMES,
    FEATURE_DIM,
    SequenceRecord,
    save_sequence,
)


def _record(
    video_id: str,
    labels: list[int],
    subject_id: str = "S1",
    scene: str = "sit",
    polarity: str = "positive",
) -> SequenceRecord:
    frames = len(labels)
    return SequenceRecord(
        features=np.zeros((FEATURE_DIM, frames), dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        valid_mask=np.ones(frames, dtype=np.float32),
        video_id=video_id,
        fps=15.0,
        subject_id=subject_id,
        scene=scene,
        polarity=polarity,
        metadata={"detection_rate": 0.9},
    )


class ActionSegmentParityTests(unittest.TestCase):
    def test_matches_the_training_time_segmentation(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            labels = rng.integers(0, len(CLASS_NAMES), size=90).astype(np.int64)
            expected = [int(labels[start]) for start, _ in _action_segments(labels)]

            self.assertEqual(action_segment_class_ids(labels), expected)

    def test_handles_edges_and_all_background(self) -> None:
        flush = np.asarray([1, 1, BACKGROUND_ID, 2, 2, 2], dtype=np.int64)
        self.assertEqual(action_segment_class_ids(flush), [1, 2])
        self.assertEqual(
            action_segment_class_ids(np.full(5, BACKGROUND_ID, dtype=np.int64)), []
        )


class GroupingTests(unittest.TestCase):
    def _tree(self, root: Path) -> None:
        save_sequence(root / "data_lm" / "a.npz", _record("a", [0, 0, BACKGROUND_ID]))
        save_sequence(root / "data_sr" / "b.npz", _record("b", [1, 1]))
        save_sequence(root / "data_vj" / "c.npz", _record("c", [2, 2]))

    def test_without_sources_each_source_is_its_own_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._tree(root)

            groups = build_groups(root)

            self.assertEqual(
                [group.label for group in groups],
                ["data_lm", "data_sr", "data_vj"],
            )
            self.assertTrue(all(len(group.sources) == 1 for group in groups))

    def test_sources_are_merged_into_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._tree(root)

            groups = build_groups(root, ["data_lm", "data_sr"])

            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].label, "data_lm+data_sr")
            self.assertEqual(groups[0].sources, ("data_lm", "data_sr"))
            self.assertEqual(len(groups[0].paths), 2)

    def test_merge_preserves_argument_order_and_drops_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._tree(root)

            groups = build_groups(root, ["data_vj", "data_lm", "data_vj"])

            self.assertEqual(groups[0].label, "data_vj+data_lm")
            self.assertEqual(len(groups[0].paths), 2)

    def test_unknown_source_lists_what_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._tree(root)

            with self.assertRaises(FileNotFoundError) as caught:
                build_groups(root, ["data_lm", "data_typo"])

            message = str(caught.exception)
            self.assertIn("data_typo", message)
            self.assertIn("data_sr", message)

    def test_falls_back_to_sequences_directly_under_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "processed"
            save_sequence(root / "loose.npz", _record("loose", [0, 0]))

            groups = build_groups(root)

            self.assertEqual([group.label for group in groups], ["processed"])

    def test_missing_or_empty_processed_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            build_groups(Path("definitely") / "not" / "here")
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(discover_sources(temp_dir), {})
            with self.assertRaises(FileNotFoundError):
                build_groups(temp_dir)


class CollectGroupStatsTests(unittest.TestCase):
    def test_counts_segments_scenes_and_shares(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            # knock twice (two runs split by background), tap once.
            save_sequence(
                root / "a.npz",
                _record("a", [0, 0, BACKGROUND_ID, 0, 1, 1]),
            )
            save_sequence(
                root / "b.npz",
                _record("b", [1, 1], subject_id="S2", scene="walk"),
            )

            stats = collect_group_stats(build_groups(root.parent)[0])

            self.assertEqual(stats.label, "data_lm")
            self.assertEqual(stats.videos, 2)
            self.assertEqual(stats.subjects, 2)
            self.assertEqual(stats.segments, 4)
            self.assertEqual(
                [(item.name, item.polarity, item.videos) for item in stats.scenes],
                [("sit", "positive", 1), ("walk", "positive", 1)],
            )

            by_name = {item.name: item for item in stats.classes}
            self.assertEqual(by_name["knock"].segments, 2)
            self.assertEqual(by_name["tap"].segments, 2)
            self.assertAlmostEqual(by_name["knock"].share, 0.5)
            self.assertAlmostEqual(sum(item.share for item in stats.classes), 1.0)

    def test_merged_group_sums_both_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_sequence(root / "data_lm" / "a.npz", _record("a", [0, 0]))
            save_sequence(
                root / "data_sr" / "b.npz",
                _record("b", [0, 0, BACKGROUND_ID, 1, 1], subject_id="S2"),
            )

            stats = collect_group_stats(build_groups(root, ["data_lm", "data_sr"])[0])

            self.assertEqual(stats.videos, 2)
            self.assertEqual(stats.subjects, 2)
            self.assertEqual(stats.segments, 3)
            by_name = {item.name: item for item in stats.classes}
            self.assertEqual(by_name["knock"].segments, 2)
            self.assertEqual(by_name["tap"].segments, 1)

    def test_classes_are_ordered_by_segment_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            labels = [1, BACKGROUND_ID, 1, BACKGROUND_ID, 1, BACKGROUND_ID, 0]
            save_sequence(root / "a.npz", _record("a", labels))

            stats = collect_group_stats(build_groups(root.parent)[0])

            self.assertEqual(
                [(item.name, item.segments) for item in stats.classes],
                [("tap", 3), ("knock", 1)],
            )

    def test_reports_absent_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            save_sequence(root / "a.npz", _record("a", [0, 0]))

            stats = collect_group_stats(build_groups(root.parent)[0])

            self.assertEqual(len(stats.missing_classes), len(CLASS_NAMES) - 2)
            self.assertNotIn("knock", stats.missing_classes)
            self.assertIn("tap", stats.missing_classes)
            self.assertNotIn("background", stats.missing_classes)

    def test_empty_group_does_not_divide_by_zero(self) -> None:
        stats = collect_group_stats(SourceGroup("empty", ("empty",), ()))

        self.assertEqual(stats.videos, 0)
        self.assertEqual(stats.segments, 0)
        self.assertEqual(stats.classes, [])
        self.assertEqual(stats.scenes, [])
        self.assertEqual(stats.polarities, {})

    def test_scenes_carry_polarity_and_are_grouped_by_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            save_sequence(root / "a.npz", _record("a", [0, 0], scene="jog"))
            save_sequence(root / "b.npz", _record("b", [0, 0], scene="jog"))
            save_sequence(root / "c.npz", _record("c", [1, 1], scene="sit"))
            save_sequence(
                root / "d.npz",
                _record(
                    "d",
                    [BACKGROUND_ID, BACKGROUND_ID],
                    scene="background-jog",
                    polarity="negative",
                ),
            )

            stats = collect_group_stats(build_groups(root.parent)[0])

            self.assertEqual(stats.polarities, {"negative": 1, "positive": 3})
            # Negative scenes first, then positive ordered by video count.
            self.assertEqual(
                [(item.name, item.polarity, item.videos) for item in stats.scenes],
                [
                    ("background-jog", "negative", 1),
                    ("jog", "positive", 2),
                    ("sit", "positive", 1),
                ],
            )

    def test_same_scene_name_under_two_polarities_stays_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            save_sequence(root / "a.npz", _record("a", [0, 0], scene="jog"))
            save_sequence(
                root / "b.npz",
                _record("b", [0, 0], scene="jog", polarity="unknown"),
            )

            stats = collect_group_stats(build_groups(root.parent)[0])

            self.assertEqual(
                [(item.name, item.polarity) for item in stats.scenes],
                [("jog", "positive"), ("jog", "unknown")],
            )


class OutputTests(unittest.TestCase):
    def test_writes_json_and_csv_covering_every_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_sequence(root / "data_lm" / "a.npz", _record("a", [0, 0, 1]))
            save_sequence(root / "data_sr" / "b.npz", _record("b", [1, 1]))
            groups = [collect_group_stats(g) for g in build_groups(root)]

            json_path, csv_path = write_stats_outputs(root / "out", groups)

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload), ["data_lm", "data_sr"])
            self.assertEqual(payload["data_lm"]["segments"], 2)
            rows = csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                rows[0], "group,class_id,name,segments,share"
            )
            self.assertEqual(len(rows), 1 + 2 + 1)

    def test_render_shows_merged_sources_classes_and_scene_polarity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_sequence(root / "data_lm" / "a.npz", _record("a", [0, 0], scene="jog"))
            save_sequence(
                root / "data_sr" / "b.npz",
                _record("b", [BACKGROUND_ID], scene="background-jog",
                        polarity="negative"),
            )
            stats = collect_group_stats(build_groups(root, ["data_lm", "data_sr"])[0])

            rendered = render_group_stats(stats)

            self.assertIn("data_lm+data_sr", rendered)
            self.assertIn("merged: data_lm, data_sr", rendered)
            self.assertIn("knock", rendered)
            self.assertIn("jog (positive)", rendered)
            self.assertIn("background-jog (negative)", rendered)
            self.assertIn("polarity: negative 1, positive 1", rendered)
            self.assertIn("absent", rendered)


if __name__ == "__main__":
    unittest.main()
