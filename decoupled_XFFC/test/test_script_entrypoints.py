import subprocess
import sys
import unittest
from pathlib import Path


class ScriptEntrypointTests(unittest.TestCase):
    def test_train_script_can_start_without_package_import_error(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_dir = repo_root / "decoupled_XFFC"
        completed = subprocess.run(
            [sys.executable, "-c", "import train"],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_process_images_entry_imports_inference_package(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_dir = repo_root / "decoupled_XFFC"
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import process_images; from XFFC.inference import load_model, process_single_image, process_folder",
            ],
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
