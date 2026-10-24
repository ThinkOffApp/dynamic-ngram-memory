"""Many users on one proxy: the users file, the keys, the owner names and each owner's store.

Each user owns their notes and has a store of their own, STORE_DIR/NAME.jsonl. A request names its owner only by its
key (Authorization: Bearer KEY). The users file keeps the SHA-256 of each key, never the key. A key is 32 random bytes
(secrets.token_urlsafe), so a hash without salt is enough: there is nothing to guess. The proxy reads the users file
once, at start, and refuses one that another account could change.

  python -m notes_proxy add-user --users users.json --owner alice --store-dir ~/notes [--reuse-notes]
  python -m notes_proxy remove-user --users users.json --owner alice --keep-notes [--store-dir ~/notes]
  python -m notes_proxy remove-user --users users.json --owner alice --erase-notes --store-dir ~/notes
"""
import contextlib
import datetime
import fcntl
import hashlib
import hmac
import json
import os
import re
import resource
import secrets
import stat
import threading

from .store import Journal, Notes, StoreError, _sync, _sync_dir

USERS_FORMAT = "notes-proxy-users/1"
NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
SHA256 = re.compile(r"[0-9a-f]{64}")
DIR_LOCK = ".notes-proxy.lock"     # held by the proxy that serves a store folder (no owner name starts with a dot)
FILES_PER_OWNER = 2                # an opened owner holds their journal's lock file and its append fd until the stop
FILES_SPARE = 64                   # the proxy's own files and sockets


class UsersError(Exception):
    """A refusal about users: a name, the users file or a store folder."""


def check_name(name):
    """The name when it is 1 to 32 of a-z, 0-9, _ and -, starting with a letter or a digit, else UsersError. So no
    dot, slash, NUL, capital (macOS folders ignore case) or letter of another script ever reaches a file name."""
    if isinstance(name, str) and name.isascii() and NAME.fullmatch(name):
        return name
    raise UsersError(f"the owner name {repr(name)[:40]} is refused: a name is 1 to 32 of a-z, 0-9, _ and -, starting "
                     f"with a letter or a digit")


def journal_path(store_dir, owner):
    """The owner's journal in the store folder: STORE_DIR/NAME.jsonl."""
    return os.path.join(os.path.abspath(store_dir), check_name(owner) + ".jsonl")


def digest(key):
    """The key's SHA-256 in hex, as the users file keeps it."""
    return hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


# ============================================================================================ the users file
def _private(path, fd):
    """UsersError when an account other than this one (or root) could change the users file: whoever can, can give
    themselves any user's notes."""
    st = os.fstat(fd)
    if st.st_uid not in (os.getuid(), 0):
        raise UsersError(f"the users file {path} belongs to another account (uid {st.st_uid}), and whoever can change "
                         f"it can read any user's notes: chown it to this account and chmod 600 it")
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise UsersError(f"other accounts can change the users file {path} (mode {stat.S_IMODE(st.st_mode):04o}), and "
                         f"whoever can change it can read any user's notes: chmod 600 {path}")


def _read(path, private=False):
    """The users file's contents, checked: every name allowed, every hash a SHA-256, no two the same. With `private`,
    also a file only this account can change."""
    try:
        with open(path, "rb") as f:
            if private:
                _private(path, f.fileno())
            data = json.loads(f.read())
    except FileNotFoundError:
        raise UsersError(f"no users file at {path}: add a user first (python -m notes_proxy add-user --users {path} "
                         f"--owner NAME)")
    except (OSError, ValueError) as e:
        raise UsersError(f"the users file {path} cannot be read: {e}")
    if not isinstance(data, dict) or data.get("format") != USERS_FORMAT or not isinstance(data.get("users"), dict):
        raise UsersError(f"{path} is not a {USERS_FORMAT} file")
    seen = set()
    for name, u in data["users"].items():
        check_name(name)
        h = u.get("sha256") if isinstance(u, dict) else None
        if not isinstance(h, str) or not SHA256.fullmatch(h):
            raise UsersError(f"the user {name} in {path} has no key hash (sha256)")
        if h in seen:
            raise UsersError(f"two users in {path} have the same key")
        seen.add(h)
    return data


@contextlib.contextmanager
def _locked(path):
    """One change to the users file at a time (PATH.lock, flock)."""
    if not os.path.isdir(os.path.dirname(path)):
        raise UsersError(f"no folder for the users file at {os.path.dirname(path)}")
    fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)                   # closing it lets the lock go


def _write(path, data):
    """The users file anew: a new file beside it (owner-only), synced, renamed over it, then the folder synced."""
    tmp = f"{path}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    raw = memoryview((json.dumps(data, indent=1, sort_keys=True) + "\n").encode())
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            while raw:
                raw = raw[os.write(fd, raw):]
            _sync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except OSError as e:
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise UsersError(f"the users file {path} could not be written ({e.strerror}), so nothing changed")
    try:
        _sync_dir(path)
    except OSError as e:
        raise UsersError(f"the users file {path} was written, but its folder could not be synced ({e.strerror})")


def add_user(path, owner, store_dir=None, reuse_notes=False):
    """A new user with a new key. Returns the key, which is kept nowhere: show it once. With `store_dir`, a name whose
    journal is still in that folder (an earlier user's, kept by remove_user) is refused unless `reuse_notes` says it
    is the same person, so a new person never reads someone else's notes by accident."""
    check_name(owner)
    path = os.path.abspath(path)
    with _locked(path):
        data = _read(path) if os.path.exists(path) else {"format": USERS_FORMAT, "users": {}}
        if owner in data["users"]:
            raise UsersError(f"{owner} is already a user: remove them first to give them a new key")
        if store_dir is not None:
            d = os.path.abspath(store_dir)
            if not os.path.isdir(d):
                raise UsersError(f"no store folder at {d}: make it first (mkdir -p {d}), or name the folder the proxy "
                                 f"serves")
            j = journal_path(d, owner)
            if os.path.lexists(j) and not reuse_notes:
                raise UsersError(f"{j} already holds notes of an earlier user named {owner}, and a new {owner} would "
                                 f"read them. If this is the same person, add them with --reuse-notes. If not, "
                                 f"choose another name, or delete that file and {owner}.jsonl.lock first")
        key = secrets.token_urlsafe(32)
        data["users"][owner] = {"sha256": digest(key), "added": datetime.date.today().isoformat()}
        _write(path, data)
    return key


def remove_user(path, owner, notes, store_dir=None):
    """Take a user out of the users file. `notes` must say what happens to their notes. "keep" leaves their journal
    in the store folder, for add_user(..., reuse_notes=True) to give back. "erase" deletes it first, which needs the
    store folder, their journal in it and no proxy serving it. Returns the erased journal's path, or None."""
    if notes not in ("keep", "erase"):
        raise UsersError('say what happens to the user\'s notes: "keep" or "erase"')
    check_name(owner)
    path = os.path.abspath(path)
    with _locked(path):
        data = _read(path)
        if owner not in data["users"]:
            raise UsersError(f"{owner} is not a user in {path}")
        erased = None
        if notes == "erase":
            if store_dir is None:
                raise UsersError("erasing a user's notes needs their store folder (--store-dir)")
            erased = erase_store(store_dir, owner)       # first, so a failure leaves the user there to try again
        del data["users"][owner]
        _write(path, data)
    return erased


# ============================================================================================ the store folder
def lock_dir(store_dir):
    """The store folder's lock (its .notes-proxy.lock, flock), held for as long as the returned fd is open."""
    d = os.path.abspath(store_dir)
    if not os.path.isdir(d):
        raise UsersError(f"no store folder at {d}")
    fd = os.open(os.path.join(d, DIR_LOCK), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise UsersError(f"a Notes Proxy is serving the store folder {d}: stop it first")
    return fd


def erase_store(store_dir, owner):
    """Delete an owner's journal and its lock file, with the folder and the journal locked, so never under a running
    proxy. Returns the journal's path. A folder that holds no journal of theirs is refused, and changed in no way: the
    folder may be the wrong one, and their notes may be elsewhere."""
    path = journal_path(store_dir, owner)
    d = os.path.dirname(path)
    missing = UsersError(f"there is no {owner}.jsonl in {d}, so nothing was erased and {owner} is still a user. Check "
                         f"--store-dir. If {owner} never had notes, remove them with --keep-notes")
    if not os.path.isdir(d):
        raise UsersError(f"no store folder at {d}, so nothing was erased and {owner} is still a user")
    if not os.path.lexists(path):
        raise missing                  # before the folder lock, so a wrong folder gets no lock file
    fd = lock_dir(store_dir)
    try:
        try:
            j = Journal(path)          # its own lock too (a one-owner proxy may serve this file), and stray rewrites go
        except StoreError as e:
            raise UsersError(str(e))
        try:
            if not os.path.lexists(path):
                raise missing          # gone while the lock was taken
            for p in (path, path + ".lock"):
                if os.path.lexists(p):
                    os.unlink(p)
            _sync_dir(path)
        except OSError as e:
            raise UsersError(f"the notes at {path} could not be erased ({e.strerror})")
        finally:
            j.close()
    finally:
        os.close(fd)
    return path


class Users:
    """The users file as it was at start: whose each key is."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.hashes = [(n, u["sha256"].encode()) for n, u in _read(self.path, private=True)["users"].items()]

    def __len__(self):
        return len(self.hashes)

    def names(self):
        return sorted(n for n, _ in self.hashes)

    def who(self, key):
        """The owner whose key this is, or None. The key's hash is compared with every user's, each in constant time,
        so the time taken does not tell whose key it was close to."""
        if not isinstance(key, str) or not key:
            return None
        d = digest(key).encode()
        found = None
        for name, h in self.hashes:
            if hmac.compare_digest(d, h):
                found = name
        return found


def make_room(n):
    """Room for n opened owners' files: the soft limit on open files raised as far as that needs (and some room for
    connections), else UsersError. Returns the limit."""
    need = FILES_PER_OWNER * n + FILES_SPARE
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    inf = resource.RLIM_INFINITY
    for want in (need + 1024, need):
        if soft != inf and soft < want:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (want if hard == inf else min(want, hard), hard))
            except (ValueError, OSError):
                continue
            soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if soft != inf and soft < need:
        raise UsersError(f"{n} users need room for {need} open files ({FILES_PER_OWNER} each and {FILES_SPARE} more), "
                         f"and this process may open only {soft} (its hard limit is "
                         f"{'unlimited' if hard == inf else hard}): raise the limit (ulimit -n) before the start, or "
                         f"serve fewer users from one proxy")
    return soft


class Owners:
    """Every user's Notes, each opened at its owner's first request for notes. One journal, one NoteStore (so one
    index and one set of n-gram weights) and one lock per owner, as with one proxy per owner. The folder is locked
    while it runs."""

    def __init__(self, store_dir, tok, config=None, abstain_line=None):
        self.store_dir = os.path.abspath(store_dir)
        try:
            os.makedirs(self.store_dir, mode=0o700, exist_ok=True)
            self.dir_fd = lock_dir(self.store_dir)
        except OSError as e:
            raise UsersError(f"no store folder at {self.store_dir} ({e.strerror})")
        self.tok, self.config, self.abstain_line = tok, config, abstain_line
        self.opened = {}               # owner -> their Notes
        self.locks = {}                # owner -> the lock their opening takes
        self.lock = threading.Lock()

    def get(self, owner):
        """The owner's Notes. A journal that refuses writes after a failed disk step (poisoned) is opened anew from
        disk, as a restart would, once no write of theirs is under way. Other owners never wait for it."""
        n = self.opened.get(owner)
        if n is not None and not n.journal.poisoned:
            return n
        with self.lock:
            lock = self.locks.setdefault(owner, threading.Lock())
        with lock:
            n = self.opened.get(owner)
            if n is not None and n.journal.poisoned:
                with n.lock:           # no write or erase of theirs is half done (and the old Notes takes no more)
                    n.close()
                del self.opened[owner]
                n = None
            if n is None:
                n = Notes(journal_path(self.store_dir, owner), owner, self.tok, config=self.config,
                          abstain_line=self.abstain_line)
                self.opened[owner] = n
        return n

    def close(self):
        with self.lock:
            for n in list(self.opened.values()):     # a copy: a late request may still open or reopen an owner
                n.close()
            self.opened = {}
            if self.dir_fd is not None:
                os.close(self.dir_fd)
                self.dir_fd = None
