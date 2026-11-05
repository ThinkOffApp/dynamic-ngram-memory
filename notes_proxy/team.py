"""The team's shared store, asked from an agent's own proxy.

An agent's proxy keeps the agent's private notes and sends its requests to the agent's own model. With --team-url it
also asks a team proxy which notes the team holds for each question, and adds them to the prompt. The team proxy may be
slow or down, and the agent's reply matters more than the team's notes, so a failed lookup costs that request its team
notes and nothing else.
"""
import http.client
import json
import os
import socket
import stat
import threading
import urllib.parse

REPLY_LIMIT = 1024 * 1024              # bytes of a lookup's reply: a block is a few hundred tokens


class TeamError(Exception):
    """Why a team lookup gave nothing. The text goes to the log and to the reply's notes field, so it never holds the
    key or the team proxy's own words."""


def read_key(path):
    """The key in `path`, refused when other accounts can read that file or it holds no key."""
    try:
        with open(path, "rb") as f:
            mode = stat.S_IMODE(os.fstat(f.fileno()).st_mode)
            raw = f.read(4096)
    except OSError as e:
        raise TeamError(f"the team key file {path} cannot be read ({e.strerror})")
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise TeamError(f"other accounts can reach the team key file {path} (mode {mode:04o}): chmod 600 {path}")
    key = raw.decode("utf-8", "replace").strip()
    if not key or not all("!" <= c <= "~" for c in key):
        raise TeamError(f"the team key file {path} holds no key")
    return key


class Team:
    def __init__(self, url, key, cap=256, timeout=2.0):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise TeamError("--team-url is an http:// or https:// base URL, such as http://team-host:8095")
        self.cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        self.host, self.port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.path = u.path.rstrip("/") + "/v1/notes/lookup"
        self.key, self.cap, self.timeout = key, cap, timeout

    def lookup(self, question, past=None):
        """The team proxy's reply for this question, or TeamError. `timeout` bounds the whole exchange, not each read
        of it, so a team proxy that sends a byte at a time cannot hold the request: the lookup runs on its own thread,
        and a late one has its socket shut, which ends the thread, so none pile up."""
        body = {"question": question, "cap": self.cap}
        if past is not None:
            body["past"] = past
        box = {}
        t = threading.Thread(target=self.ask, args=(body, box), daemon=True, name="team-lookup")
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            self.hang_up(box)
            raise TeamError("timed out")
        if "failed" in box:
            raise TeamError(box["failed"])
        return box["reply"]

    @staticmethod
    def hang_up(box):
        sock = box.get("sock")
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def ask(self, body, box):
        try:
            box["reply"] = self.exchange(body, box)
        except TeamError as e:
            box["failed"] = str(e)

    def exchange(self, body, box):
        conn = self.cls(self.host, self.port, timeout=self.timeout)
        try:
            conn.connect()
            box["sock"] = conn.sock    # kept here: the connection lets go of it once a reply that ends with it starts
            conn.request("POST", self.path, json.dumps(body, ensure_ascii=False).encode(),
                         {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            r = conn.getresponse()
            data = r.read(REPLY_LIMIT + 1)
        except socket.timeout:
            raise TeamError("timed out")
        except (OSError, http.client.HTTPException) as e:
            raise TeamError(type(e).__name__)
        finally:
            conn.close()
        if r.status != 200:
            raise TeamError(f"HTTP {r.status}")
        if len(data) > REPLY_LIMIT:
            raise TeamError("the reply is too long")
        try:
            v = json.loads(data)
        except ValueError:
            raise TeamError("the reply is not JSON")
        if not isinstance(v, dict) or not isinstance(v.get("block"), str) or not isinstance(v.get("ids"), list):
            raise TeamError("the reply is not a lookup")
        return v
