"""Store admission must refuse unsafe existing files without repairing or reading them."""
import os
import tempfile
import unittest
from unittest import mock

from notes_proxy.store import Notes, StoreError
from notes_proxy.tests import helpers as H


class StoreAdmission(unittest.TestCase):
    def test_oversize_rewrite_keeps_old_journal_and_cleans_temporary(self):
        from notes_proxy import store
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            path = os.path.join(root, "owner.jsonl")
            n = Notes(path, "owner", H.tokenizer())
            try:
                with open(path, "rb") as f:
                    before = f.read()
                with mock.patch.object(store, "JOURNAL_READ_LIMIT", 1):
                    with self.assertRaises(StoreError):
                        n.journal.rewrite(n.header, [])
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), before)
                self.assertFalse(any(".tmp." in name for name in os.listdir(root)))
                self.assertFalse(n.journal.poisoned)
            finally:
                n.close()

    def test_read_limit_refuses_before_truncating_torn_tail(self):
        from notes_proxy import store
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            path = os.path.join(root, "owner.jsonl")
            with open(path, "wb") as f:
                f.write(b"x" * 33)
            os.chmod(path, 0o600)
            with mock.patch.object(store, "JOURNAL_READ_LIMIT", 32, create=True):
                with self.assertRaises(StoreError):
                    Notes(path, "owner", H.tokenizer())
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"x" * 33)

    def test_write_limit_refuses_without_disk_or_memory_change(self):
        from notes_proxy import store
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            path = os.path.join(root, "owner.jsonl")
            n = Notes(path, "owner", H.tokenizer())
            try:
                with open(path, "rb") as f:
                    before = f.read()
                with mock.patch.object(store, "JOURNAL_WRITE_LIMIT", len(before), create=True):
                    with self.assertRaises(StoreError) as error:
                        n.write_text("My aunt Brigitta grows tomatoes.")
                self.assertEqual(error.exception.status, 507)
                self.assertEqual(n.listing(), [])
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), before)
            finally:
                n.close()

    def test_unsafe_lock_is_refused_and_not_chmodded(self):
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            path = os.path.join(root, "owner.jsonl")
            lock = path + ".lock"
            with open(lock, "wb") as f:
                f.write(b"sentinel")
            os.chmod(lock, 0o644)
            with self.assertRaises(StoreError):
                Notes(path, "owner", H.tokenizer())
            self.assertFalse(os.path.exists(path))
            self.assertEqual(os.stat(lock).st_mode & 0o777, 0o644)
            with open(lock, "rb") as f:
                self.assertEqual(f.read(), b"sentinel")

    def test_fifo_journal_is_refused_without_waiting_for_writer(self):
        with tempfile.TemporaryDirectory(prefix="notes-admission-") as root:
            path = os.path.join(root, "owner.jsonl")
            os.mkfifo(path, 0o600)
            with self.assertRaises(StoreError):
                Notes(path, "owner", H.tokenizer())

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
