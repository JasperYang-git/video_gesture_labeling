from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.data_loader import _action_segments
from utils.preprocessing.stats import (
    DEFAULT_WINDOW_SIZE,
    SourceGroup,
    action_segment_class_ids,
    action_segment_spans,
    build_groups,
    collect_group_stats,
    discover_sources,
    nearest_rank,
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


def _segments(pieces: list[tuple[int, int]]) -> list[int]:
    """Flatten ``(label, repeat)`` pairs into a frame label sequence."""
    return [label for label, repeat in pieces for _ in range(repeat)]


class ActionSegmentParityTests(unittest.TestCase):
    def test_matches_the_training_time_segmentation(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            labels = rng.integers(0, len(CLASS_NAMES), size=90).astype(np.int64)
            expected = [int(labels[start]) for start, _ in _action_segments(labels)]

            self.assertEqual(action_segment_class_ids(labels), expected)

    def test_spans_match_the_training_time_boundaries(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(50):
            labels = rng.integers(0, len(CLASS_NAMES), size=90).astype(np.int64)
            expected = [
                (int(labels[start]), start, end)
                for start, end in _action_segments(labels)
            ]

            self.assertEqual(action_segment_spans(labels), expected)

    def test_handles_edges_and_all_background(self) -> None:
        flush = np.asarray([1, 1, BACKGROUND_ID, 2, 2, 2], dtype=np.int64)
        self.assertEqual(action_segment_class_ids(flush), [1, 2])
        self.assertEqual(action_segment_spans(flush), [(1, 0, 2), (2, 3, 6)])
        self.assertEqual(
            action_segment_class_ids(np.full(5, BACKGROUND_ID, dtype=np.int64)), []
        )


class NearestRankTests(unittest.TestCase):
    def test_picks_the_nearest_rank_element(self) -> None:
        values = [10, 20, 30, 40]

        self.assertEqual(nearest_rank(values, 0.5), 20)
        self.assertEqual(nearest_rank(values, 0.9), 40)
        self.assertEqual(nearest_rank(values, 0.25), 10)

    def test_degenerate_inputs_stay_in_range(self) -> None:
        self.assertEqual(nearest_rank([], 0.5), 0.0)
        self.assertEqual(nearest_rank([7], 0.99), 7)
        self.assertEqual(nearest_rank([1, 2, 3], 0.0), 1)
        self.assertEqual(nearest_rank([1, 2, 3], 1.0), 3)


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
        self.assertEqual(stats.gap_percentiles, {})
        self.assertEqual(stats.window_size, DEFAULT_WINDOW_SIZE)


class SegmentLengthStatsTests(unittest.TestCase):
    def _stats(self, labels: list[int], window_size: int = DEFAULT_WINDOW_SIZE):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            save_sequence(root / "a.npz", _record("a", labels))

            return collect_group_stats(build_groups(root.parent)[0], window_size)

    def test_reports_length_percentiles_and_totals(self) -> None:
        # knock segments of 10, 20, 30 and 40 frames, split by background.
        labels = _segments(
            [
                (0, 10), (BACKGROUND_ID, 5),
                (0, 20), (BACKGROUND_ID, 5),
                (0, 30), (BACKGROUND_ID, 5),
                (0, 40),
            ]
        )

        knock = self._stats(labels, window_size=25).classes[0]

        self.assertEqual(knock.name, "knock")
        self.assertEqual(knock.segments, 4)
        self.assertEqual(knock.frames_total, 100)
        self.assertAlmostEqual(knock.mean_frames, 25.0)
        self.assertEqual(knock.p50_frames, 20)
        self.assertEqual(knock.p90_frames, 40)
        self.assertEqual(knock.p95_frames, 40)
        self.assertEqual(knock.p99_frames, 40)
        self.assertEqual(knock.max_frames, 40)
        # fps is 15, so a 20-frame segment lasts 1.333 s.
        self.assertAlmostEqual(knock.p50_sec, 20 / 15)
        self.assertAlmostEqual(knock.p95_sec, 40 / 15)

    def test_over_window_excludes_segments_exactly_the_window_length(self) -> None:
        # 29 fits, 30 exactly fills the window, only 31 forces the truncating
        # fallback branch in _dynamic_action_windows_for_record.
        labels = _segments(
            [
                (0, 29), (BACKGROUND_ID, 5),
                (0, 30), (BACKGROUND_ID, 5),
                (0, 31),
            ]
        )

        knock = self._stats(labels, window_size=30).classes[0]

        self.assertEqual(knock.over_window, 1)
        self.assertAlmostEqual(knock.over_window_share, 1 / 3)

    def test_starts_p50_counts_available_window_offsets(self) -> None:
        labels = _segments([(0, 20), (BACKGROUND_ID, 5), (0, 20)])

        knock = self._stats(labels, window_size=60).classes[0]

        # [segment_end - 60, segment_start] spans 60 - 20 + 1 = 41 offsets.
        self.assertEqual(knock.starts_p50, 41)

    def test_starts_p50_floors_at_zero_for_oversized_segments(self) -> None:
        labels = _segments([(0, 90)])

        knock = self._stats(labels, window_size=60).classes[0]

        self.assertEqual(knock.starts_p50, 0)
        self.assertEqual(knock.over_window, 1)

    def test_window_size_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            collect_group_stats(SourceGroup("empty", ("empty",), ()), 0)

    def test_gaps_measure_background_between_neighbouring_actions(self) -> None:
        labels = _segments(
            [
                (0, 5), (BACKGROUND_ID, 12),
                (1, 5), (BACKGROUND_ID, 4),
                (0, 5), (BACKGROUND_ID, 8),
                (1, 5),
            ]
        )

        stats = self._stats(labels)

        # Gaps are 12, 4 and 8; sorted that is [4, 8, 12].
        self.assertEqual(stats.gap_percentiles, {"p05": 4, "p25": 4, "p50": 8})

    def test_touching_actions_of_different_classes_have_a_zero_gap(self) -> None:
        stats = self._stats(_segments([(0, 5), (1, 5)]))

        self.assertEqual(stats.gap_percentiles, {"p05": 0, "p25": 0, "p50": 0})

    def test_gaps_do_not_span_across_videos(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            pieces = _segments([(0, 5), (BACKGROUND_ID, 6), (0, 5)])
            save_sequence(root / "a.npz", _record("a", pieces))
            save_sequence(root / "b.npz", _record("b", pieces, subject_id="S2"))

            stats = collect_group_stats(build_groups(root.parent)[0])

            # Two videos with one gap each; no phantom third gap between them.
            self.assertEqual(stats.segments, 4)
            self.assertEqual(stats.gap_percentiles, {"p05": 6, "p25": 6, "p50": 6})

    def test_window_size_is_recorded_on_the_group(self) -> None:
        stats = self._stats(_segments([(0, 5)]), window_size=96)

        self.assertEqual(stats.window_size, 96)

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
                rows[0],
                "group,class_id,name,segments,share,frames_total,mean_frames,"
                "p50_frames,p90_frames,p95_frames,p99_frames,max_frames,"
                "p50_sec,p95_sec,over_window,over_window_share,starts_p50",
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

    def test_render_shows_length_columns_and_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data_lm"
            labels = _segments([(0, 4), (BACKGROUND_ID, 9), (0, 4)])
            save_sequence(root / "a.npz", _record("a", labels))
            stats = collect_group_stats(build_groups(root.parent)[0])

            rendered = render_group_stats(stats)

            self.assertIn("window 60", rendered)
            self.assertIn("p95f", rendered)
            self.assertIn("starts50", rendered)
            self.assertIn("gap to next action (frames): p05 9, p25 9, p50 9", rendered)


if __name__ == "__main__":
    unittest.main()
