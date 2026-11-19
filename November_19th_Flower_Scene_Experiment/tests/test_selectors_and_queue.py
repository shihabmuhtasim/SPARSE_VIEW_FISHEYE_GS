"""CPU checks of selection behavior and queue safeguards, not GPU training."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "code"))
from prepare_study import fps, coverage
from run_experiments import command, inventory, preserve_partial, verify_finished


class Selectors(unittest.TestCase):
    def test_angular_fps_uses_full_3d_directions(self):
        cameras = np.array([[1., 0., 0.], [0., 0., 1.], [-1., 0., 0.], [0., 0., -1.]])
        chosen, history, _, _ = fps(cameras, np.zeros(3), 4)
        self.assertEqual(chosen, [0, 2, 1, 3])
        self.assertAlmostEqual(history[1]["nearest_selected_angle_at_addition"], 180.)
        self.assertAlmostEqual(history[2]["nearest_selected_angle_at_addition"], 90.)

    def test_coverage_rewards_new_directions_without_new_point_ids(self):
        cameras = np.array([[1., 0., 0.], [2., 0., 0.], [-1., 0., 0.]])
        chosen, history, visible, _ = coverage(cameras, np.zeros((1, 3)), [[0, 1, 2]], 3)
        self.assertEqual(chosen, [0, 2, 1])
        self.assertEqual([h["new_points"] for h in history], [1, 0, 0])
        self.assertGreater(history[1]["directional_gain"], 0.)
        self.assertTrue(visible.all())

    def test_frozen_manifests_and_test_exclusion(self):
        folder = PACKAGE / "frozen/flowers"
        base = json.loads((folder / "flowers_fullview_split_v1.json").read_text())
        self.assertEqual(len(base["train_images"]), 130)
        self.assertEqual(len(base["test_images"]), 19)
        for method in ["pca_angle", "angular_fps", "coverage"]:
            previous = []
            for count in [12, 24, 48, 96]:
                item = json.loads((folder / f"flowers_{method}_{count}_v1.json").read_text())
                self.assertEqual(len(set(item["train_images"])), count)
                self.assertEqual(item["test_images"], base["test_images"])
                self.assertFalse(set(item["train_images"]) & set(base["test_images"]))
                if method != "pca_angle":
                    self.assertEqual(item["train_images"][:len(previous)], previous)
                previous = item["train_images"]


class Queue(unittest.TestCase):
    def test_training_settings_match_executed_run(self):
        args = command(Path("source"), Path("scene"), Path("run"), Path("split.json"))
        for flag, value in [("--bs", "2"), ("--cvo_preset", "recip"), ("--iterations", "30000"), ("--seed", "0")]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--eval", args)
        self.assertIn("--disable_viewer", args)
        self.assertNotIn("--resolution", args)
        self.assertNotIn("--start_checkpoint", args)

    def test_partial_attempt_preserved_and_finished_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "runs/pca_angle_12"
            (run / "model").mkdir(parents=True)
            (run / "model/partial.txt").write_text("original interrupted evidence")
            old = preserve_partial(run)
            self.assertEqual((old / "model/partial.txt").read_text(), "original interrupted evidence")
            self.assertFalse((run / "model").exists())
            proof = run / "result.json"
            proof.write_text('{"PSNR":17.0}')
            hashes = inventory(run)
            (run / "completed_sha256.json").write_text(json.dumps(hashes))
            (run / "done.json").write_text('{"state":"complete"}')
            self.assertTrue(verify_finished(run))
            proof.write_text("changed")
            with self.assertRaises(AssertionError):
                verify_finished(run)


if __name__ == "__main__":
    unittest.main()
