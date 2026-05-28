import unittest
from unittest import mock

from decoupled_yffc import train as train_entry
from decoupled_yffc.yffc import cli


class TrainingEntrypointTests(unittest.TestCase):
    def test_decoupled_train_entry_uses_internal_engine(self):
        cfg = {}

        with mock.patch("decoupled_yffc.train.train_model") as mocked_train_model:
            mocked_train_model.return_value = {"ok": True}
            with mock.patch("decoupled_yffc.train.datetime") as mocked_datetime:
                mocked_datetime.now.return_value.strftime.return_value = "20260101_000000"
                with mock.patch("builtins.print"):
                    train_entry.main()

        mocked_train_model.assert_called_once()
        passed_cfg = mocked_train_model.call_args.args[0]
        self.assertEqual(passed_cfg["OUTPUT_FOLDER"], "experiments/yffc_v2_20260101_000000")
        self.assertFalse(passed_cfg["USE_AMP"])

    def test_cli_train_routes_to_internal_engine(self):
        argv = [
            "prog",
            "train",
            "--input_dir",
            "in",
            "--target_dir",
            "tg",
            "--epochs",
            "3",
            "--batch_size",
            "2",
            "--image_size",
            "64",
            "--save_dir",
            "out",
            "--save_interval",
            "1",
        ]

        with mock.patch("sys.argv", argv):
            with mock.patch("decoupled_yffc.yffc.cli.train_model") as mocked_train_model:
                mocked_train_model.return_value = {"ok": True}
                cli.main()

        mocked_train_model.assert_called_once()
        passed_cfg = mocked_train_model.call_args.args[0]
        self.assertEqual(passed_cfg["INPUT_DIR"], "in")
        self.assertEqual(passed_cfg["TARGET_DIR"], "tg")
        self.assertEqual(passed_cfg["EPOCHS"], 3)
        self.assertEqual(passed_cfg["BATCH_SIZE"], 2)
        self.assertEqual(passed_cfg["IMAGE_SIZE"], (64, 64))
        self.assertEqual(passed_cfg["OUTPUT_FOLDER"], "out")


if __name__ == "__main__":
    unittest.main()
