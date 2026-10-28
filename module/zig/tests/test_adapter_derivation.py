"""Provenance guard for the forthcoming versioned native adapter namespace."""
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("derive_adapter", ROOT / "tools/derive_notes_adapter.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Derivation(unittest.TestCase):
    def test_deterministic_copy_and_tamper_detection(self):
        with tempfile.TemporaryDirectory(prefix="notes-derive-") as root:
            output = pathlib.Path(root).resolve() / "adapter"
            module.derive(output)
            module.derive(output, verify=True)
            target = output / "server/notes/journal.zig"
            # Deliberate corruption is generated test data, not the pinned export.
            target.write_bytes(target.read_bytes() + b"\n// tamper\n")
            with self.assertRaisesRegex(ValueError, "generated source mismatch"):
                module.derive(output, verify=True)

    def test_unmanaged_output_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="notes-derive-guard-") as root:
            output = pathlib.Path(root).resolve()
            sentinel = output / "user-file"
            sentinel.write_text("keep")
            with self.assertRaisesRegex(ValueError, "non-generated directory"):
                module.derive(output)
            self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
