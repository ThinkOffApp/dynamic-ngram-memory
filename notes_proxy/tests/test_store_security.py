"""Store admission must refuse unsafe existing files without repairing or reading them."""
import os
import tempfile
import unittest

from notes_proxy.store import Notes, StoreError
from notes_proxy.tests import helpers as H


class StoreAdmission(unittest.TestCase):
    def test_world_readable_journal_is_refused_without_changing_it(self):
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            path = os.path.join(root, "owner.jsonl")
            n = Notes(path, "owner", H.tokenizer())
            n.write_text("My aunt Brigitta grows tomatoes.")
            n.close()
            os.chmod(path, 0o644)
            with open(path, "rb") as f:
                before = f.read()
            opened = None
            try:
                with self.assertRaises(StoreError):
                    opened = Notes(path, "owner", H.tokenizer())
            finally:
                if opened is not None:
                    opened.close()
            with open(path, "rb") as f:
                self.assertEqual(f.read(), before)
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o644)

    def test_journal_symlink_is_refused_and_target_unchanged(self):
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            target = os.path.join(root, "target.jsonl")
            path = os.path.join(root, "owner.jsonl")
            n = Notes(target, "owner", H.tokenizer())
            n.close()
            with open(target, "rb") as f:
                before = f.read()
            os.symlink(target, path)
            opened = None
            try:
                with self.assertRaises(StoreError):
                    opened = Notes(path, "owner", H.tokenizer())
            finally:
                if opened is not None:
                    opened.close()
            self.assertTrue(os.path.islink(path))
            with open(target, "rb") as f:
                self.assertEqual(f.read(), before)
