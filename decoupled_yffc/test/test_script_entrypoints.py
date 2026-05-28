import subprocess
import sys
import unittest
from pathlib import Path


class ScriptEntrypointTests(unittest.TestCase):
    def test_train_script_can_start_without_package_import_error(self):
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "decoupled_yffc" / "train.py"

        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        self.assertNotIn("ModuleNotFoundError: No module named 'decoupled_yffc'", completed.stderr)


if __name__ == "__main__":
    unittest.main()
