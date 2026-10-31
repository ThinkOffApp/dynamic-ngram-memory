# Notes Proxy

**TensorFold Notes for any OpenAI-compatible inference server.**

The Notes Proxy is a small server that sits between your chat client and the server that runs the model: llama.cpp's `llama-server`, vLLM, Ollama, LM Studio, SGLang or TensorFold itself. When a chat request passes through it, the proxy looks up what this user has told it before (their notes), puts the notes that fit the question into the request as the first system message, and forwards the request. The model then answers with the notes in front of it. Nothing about the model or its server changes.

The lookup is not a copy. The proxy calls the test suite's own note memory, [`tests/zengine/notes.py`](../tests/zengine/notes.py): the same n-gram matching, scoring, hop, 512-token cap and framing that the paper measures and that the TensorFold server's lookup was checked against. For every question the proxy sends exactly the block `NoteStore.retrieve()` returns, byte for byte (tested below), unless you change its first line with `--framing` or `--abstain-line`.

## Run it

You need Python 3.10 or later and the `tokenizers` package (`pip install tokenizers`), and the model's `tokenizer.json`. The 512-token cap is counted with the model's own tokenizer, as in the paper. Run the commands from this repository's top folder.

The notes are kept in a folder of your own. Make it first, readable by you only:

```sh
mkdir -p -m 700 ~/notes
```

In front of llama.cpp (`llama-server -m model.gguf --port 8080` running). A GGUF file has no `tokenizer.json`: use the one from the model's original Hugging Face repository.

```sh
python3 -m notes_proxy --upstream http://127.0.0.1:8080 --owner me --store ~/notes/me.jsonl --tokenizer ~/models/my-model/tokenizer.json
```

In front of vLLM (`vllm serve MODEL --port 8000` running):

```sh
python3 -m notes_proxy --upstream http://127.0.0.1:8000 --owner me --store ~/notes/me.jsonl --tokenizer ~/.cache/huggingface/hub/models--ORG--MODEL/snapshots/HASH/tokenizer.json
```

The proxy can start TensorFold itself (`--tensorfold` instead of `--upstream`). The child is bound to 127.0.0.1, the API key is in `TENSORFOLD_API_KEY` (never argv), token counts go through the child's `/v1/tokenize` and piece text through `/v1/detokenize`, and an erase restarts TensorFold once the proxy is idle, after a quiet window (`--tensorfold-quiet`, default 10 s), so several erases share one reload. SIGTERM, SIGHUP and process exit stop the child. If traffic never goes idle, new requests are refused after `--tensorfold-restart-max` (default 60 s) so the restart still happens. `GET /v1/notes/erase-status` turns pending into complete when that restart has landed. Extra serve flags must be on the allowlist, written with equals so argparse does not eat them: `--tensorfold-arg=--no-drafts`. `--tensorfold` with `--users` is refused unless you also pass `--share-upstream-cache`: TensorFold has no per-user prompt-cache isolation, so users would share the cache.

```sh
python3 -m notes_proxy --tensorfold org/name --owner me --store ~/notes/me.jsonl
```

Then point your chat client at `http://127.0.0.1:8090/v1` instead of the server's own address. Add a note and ask about it:

```sh
curl -s http://127.0.0.1:8090/v1/notes -H "Content-Type: application/json" -d '{"text": "My sister Anna drives a red Volvo."}'
```

```sh
curl -s http://127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "m", "messages": [{"role": "user", "content": "What car does Anna drive?"}]}'
```

The reply is the server's own, with one more field, `notes`: the ids of the notes given (see Note ids), their tokens, and whether the question was read as one about the past.

To try it on a test set's notes, make a store from one of the generated sets first:

```sh
python3 -m notes_proxy seed --set tests/examples/L6-seed1.json --phase teach --owner user --store /tmp/l6.jsonl
```

## What it does

- **`POST /v1/chat/completions`.** The last user message is the question (a string, or its text parts joined, as the TensorFold server reads it). If the lookup finds notes, the block goes in first as `{"role": "system", "content": block}`; the client's own messages follow it unchanged. If it finds nothing, nothing is put in. Then the request goes to the upstream server.
- **Streaming.** A streamed reply is passed on line by line as the upstream sends it, never held back. The `notes` object comes as an SSE comment line, `: notes {...}`, just before `data: [DONE]`. Every SSE parser skips comment lines, so no client meets a chunk it does not expect. The same ids, tokens and past flag are also in the reply headers `X-Notes-Ids`, `X-Notes-Tokens` and `X-Notes-Past`. A reply that is not streamed gets the `notes` field added at the end of its JSON; the upstream's bytes before it are left as they were.
- **Request fields**, as the TensorFold server reads them: `"notes": false` sends the request on without notes; `"notes_past": true/false` forces the view of the past (absent: the question decides); `"notes_exclude": ["w12", ...]` leaves the notes with those ids out; `"notes_in_weights": true` asks for the notes the pocket holds to ride the block again (a control arm: by default a taught note is left out, since the weights answer it alone); `"notes_cap": n` sets the cap; `"notes_echo": true` puts the block itself in the reply; `"notes_write": false` keeps this message out of the notes. A field of the wrong type is a 400. The proxy removes these fields before forwarding, so the upstream never sees them.
- **Every other route** (`/v1/models`, `/v1/completions`, `/v1/embeddings` and the rest) is passed through as it is. With many users only these three pass (see Many users).
- **The upstream's API key**, if it needs one, is the client's: the `Authorization` header is passed on as sent. With `--upstream-key-env NAME` the proxy sends the key held in the environment variable `NAME` instead. With many users the client's key is never passed on (see Many users). The proxy never stores a key and never logs one. It logs one line per request: the route, the status, the note ids and times, never a message or a note's text.

## Notes API

| Request | What it does |
| --- | --- |
| `GET /v1/notes` | the current notes; `?history=1` also lists the replaced and taken-back ones |
| `GET /v1/notes/ID` | one note |
| `POST /v1/notes` `{"text", "subject"?}` | a new note, through the intake check (201 when written, 200 when it was already known) |
| `DELETE /v1/notes/ID` | erase: the note is removed from the store, its journal and every cache the proxy holds, as if it had never been said |
| `GET /v1/notes/erase-status` | with `--tensorfold`: whether a restart newer than the last erase has finished |
| `POST /v1/notes/ID/weights` | with `--tensorfold`: mark the note taught (`{"in_weights": true}`) or open again (`false`, optional `reason`). A mark true is accepted only when the child's own graph shows a learned fact with this note's exact text (a claim of a lesson never taught would otherwise retire the note from the prompt — it is a 409). Only the proxy's operator/promoter calls this; see Living Weights below |
| `GET /v1/slide/graph` | with `--tensorfold` and one owner: the child's Living Weights graph, read-only, with each fact mapped to `note` (the id of your note whose text is the taught sentence, or null) and marked `in_weights`. The child's own fact ids, states and questions come through as they are. Refused (404) with many users: the pocket is shared, so its whole graph — every user's taught sentences and questions — would go to whoever asks |

An erase also removes the writes that were kept out as the same as that note, and the take-backs of it. With `--upstream` the erase reply always says `"complete": false`: the proxy cannot reach the upstream server's prompt cache or logs, and Python does not overwrite memory it frees. With `--tensorfold` the DELETE replies `"pending"` at once (it does not wait for the model to reload); `GET /v1/notes/erase-status` says `"complete": true` once a restart newer than that erase has finished. Python still does not overwrite memory it frees.

**Note ids.** A note's id is the id of the write that made it: `w1`, `w2` and so on for notes written through the proxy, and `s1`, `s2` and so on for the lessons a store was seeded with. An id names one note for good. No erase, restart or later write gives it to another note. So an erase sent twice, for example by a client that retried after a timeout, erases the note once. The second reply is a 404, and nothing changes.

The block itself numbers its notes `[1]`, `[2]` and so on, as `notes.py` does, and those numbers do move: after an erase, every later note's number goes down by one. So each note in `GET /v1/notes` has its `number` next to its `id`, the chat reply's `notes` has `numbers` next to `ids` in the same order, and the erase reply's `renumbered` lists the numbers that moved, with each note's id. Use the id, never the number, to name a note.

**The id format changed.** Before, a note's id was its number. `GET /v1/notes/3`, `DELETE /v1/notes/3`, `"notes_exclude": [3]`, the reply's `notes.ids` and `X-Notes-Ids` all used numbers, and an erase moved them. A repeated erase then removed a different note, and that cannot be undone. Now ids are text, such as `"w3"`. A number where an id goes is a 400 and changes nothing. A client that kept numbers must read the ids from `GET /v1/notes` again. The store file did not change: a store made before opens as it is, and its ids come from the writes it already holds. A store in which a write has no id, or two writes share one, is refused at start (the proxy never writes one).

The notes are kept in the `--store` file, a journal with one write per line. Each write is on disk (synced) before it is answered or used. At startup the journal is replayed, so a restart keeps every note with the same ids. An erase writes the journal anew without the erased writes. The file holds the notes in plain text: keep it where only you can read it (the proxy makes it readable by its owner only).

## Writing notes from chat (`--write`, off by default)

With `--write`, the proxy writes notes from what the user says, by the TensorFold server's rule:

- only the conversation's last message, and only when it is the user's own; never a system message, an earlier message or the model's answer (an answer can be wrong, and storing it would keep the mistake);
- only a statement: a message that reads as a question ("Where do I live?", "what car does Anna drive") or a request ("Tell me...", "Remind me...") is skipped, as is anything over one line or 400 bytes. The question check is the server's own (English only), ported to Python;
- through the intake check, after the lookup, before the request is forwarded: the next request finds the note.

A sentence in the wordings the rule book reads ("My favourite colour is green.", "I live in Oslo now.") goes through the rule book's intake check (`intake.classify`): new, replaces an earlier note, already known, or implied. Any other sentence is written as the TensorFold server writes every chat statement: already known if a current note of the same subject says the same words, otherwise new. The reply's `notes.write` says what happened, or why the message was skipped.

## Many users

One proxy can serve many people. Each person is a user with a key of their own. The key decides whose notes a request reads and writes. The client never names the owner.

Make a folder for the users file and the notes, readable by you only:

```sh
mkdir -p -m 700 ~/notes
```

Add each user to a users file. The command prints their new key once, on its own line. Give it to that person.

```sh
python3 -m notes_proxy add-user --users ~/notes/users.json --owner alice --store-dir ~/notes
```

Start the proxy with the users file and the notes folder, in place of `--owner` and `--store`:

```sh
python3 -m notes_proxy --upstream http://127.0.0.1:8080 --users ~/notes/users.json --store-dir ~/notes --tokenizer ~/models/my-model/tokenizer.json
```

Each request sends its user's key. An OpenAI client sends it when its API key is set to that key.

```sh
curl -s http://127.0.0.1:8090/v1/notes -H "Authorization: Bearer KEY" -H "Content-Type: application/json" -d '{"text": "My sister Anna drives a red Volvo."}'
```

- **One store per user.** Each user's notes are in `STORE_DIR/NAME.jsonl`, opened at that user's first request for notes. Each store has its own index, its own n-gram weights and its own lock, as with one proxy per user. One user's notes never reach another user's block, and never change another user's scores. The upstream is where they could still meet, so the proxy keeps apart there what it can (see below, and What this proxy cannot keep apart).
- **A missing or unknown key is a 401**, before any note is looked up and before anything goes to the upstream. Every route needs a key, also the ones passed through. The one exception is a browser's CORS preflight (`OPTIONS` with an allowed `Origin`), which never carries a key. Keys are compared by their SHA-256, in constant time.
- **The key never goes to the upstream.** The proxy drops it from every header. A request that also has the key in its URL or in its body is refused with a 400. The upstream gets the key in the environment variable that `--upstream-key-env NAME` names, or no `Authorization` at all.
- **`/v1/notes` and `--write` act on the key's own user only.** A note id is that user's own: alice's `w1` and bob's `w1` are two notes, and each key reaches only its own.
- **Only the model routes pass through.** These are `GET /v1/models`, `POST /v1/completions` and `POST /v1/embeddings`, besides `POST /v1/chat/completions`. Any other route is a 404. So no user reaches the upstream's own pages, such as llama.cpp's `/slots`, `/metrics` and `/props`, which can show or change other users' prompts.
- **The upstream's prompt cache is kept apart.** Every block starts with the same framing. An upstream that reuses the start of one prompt for the next would let a user test guesses about another user's notes: send the framing and a guess, then see how much of the prompt was reused. So the proxy sends every chat, completion and embedding request with `"cache_prompt": false`, which tells llama.cpp to reuse no cached prompt. It also sends a `"cache_salt"` of the user's own, which tells vLLM to reuse only that user's cache. Whatever the client sent for these two is replaced. Replies lose llama.cpp's `timings` and `prompt_progress`, and the `cached_tokens` count in `usage`, because these count the reused tokens. With the cache off, llama.cpp reads the whole prompt every time, so long chats are slower. `--share-upstream-cache` turns all of this off. Use it only when every user may know what the others' notes say.
- **Other servers may ignore both fields.** The proxy then cannot keep the users' caches apart, and a user could still time the replies. Run one upstream per user there, or turn its prefix cache off.
- **No client names another user's state on the upstream.** A field such as `id_slot` or `previous_response_id`, or a header such as `X-Session-Id`, could point two users at one slot, session or saved conversation. The table at the end of this section says what the proxy does with each.
- **The key alone names the owner.** A field or header that names one, such as `owner` or `X-Owner`, never changes whose notes a request reads.
- **Three of the client's headers go on: `Content-Type`, `Accept` and `User-Agent`.** The proxy drops every other one. So no cookie, no other key (`api-key`, `X-Api-Key`) and no session or project header reaches the upstream.
- **No times in the replies.** The reply's `notes` object has no `lookup_us`, `total_us` or `write_us`. All users share one tokenizer, and its cache makes a word that another user used a little faster to look up. The operator's log keeps the times.
- **The users file keeps no key.** It keeps the SHA-256 of each key. A lost key cannot be shown again. Remove the user with `--keep-notes`, then add them again with `--reuse-notes` for a new key to the same notes.
- **A name passes its notes on only when you say so.** `add-user` refuses a name whose `NAME.jsonl` is still in the store folder, unless `--reuse-notes` says it is the same person. So a new user never reads an earlier user's notes by accident.
- **Names** are 1 to 32 of `a-z`, `0-9`, `_` and `-`, starting with a letter or a digit. Any other name is refused when the user is added and when the users file is read. So no name can point outside the folder. A name has no capital letters, so two names never share a file, even on a disk that ignores case. The file is named by the name itself, not by a hash of it, so you can see whose notes each file holds.
- **The users file is read once, at start.** Restart the proxy after `add-user` or `remove-user`. The proxy refuses to start when another account could change the file, because whoever can change it can read any user's notes. `add-user` writes it readable by you only.
- **Two open files per user.** Each user whose notes are open holds two open files until the proxy stops. At start the proxy raises its limit on open files to fit every user, and refuses to start when it cannot (see `ulimit -n`).
- **A store that cannot be opened** gives its own user a 503 or a 500, and the cause goes to the operator's log. When a disk step fails and a journal stops taking writes, that user's next request opens it again from disk, as a restart would. The other users are not affected.

Removing a user takes `--keep-notes` or `--erase-notes`, one of the two:

```sh
python3 -m notes_proxy remove-user --users ~/notes/users.json --owner alice --keep-notes
```

```sh
python3 -m notes_proxy remove-user --users ~/notes/users.json --owner alice --erase-notes --store-dir ~/notes
```

`--keep-notes` leaves `alice.jsonl` in the folder, for `add-user --reuse-notes` to give back. `--erase-notes` deletes that file. It refuses while a proxy serves the folder, so stop the proxy first. It also refuses, and keeps the user, when the folder holds no `alice.jsonl`, because the folder may be the wrong one. As with any deleted file, the disk may keep the old bytes until they are written over.

A key travels in clear over plain HTTP. Keep the proxy on 127.0.0.1, or put it behind TLS before you use `--allow-remote`.

### What reaches the upstream

An upstream may keep state from one request to the next: a cached prompt, a slot, a session or a saved conversation. A request can name that state by a field in its body or by a header. Two users who sent the same name would share that state. So with many users the proxy does this with each such field, in chat, completion and embedding requests alike:

| Field | In | With many users |
| --- | --- | --- |
| `cache_prompt` | body | set to `false`, so llama.cpp reuses no cached prompt |
| `cache_salt` | body | replaced by the user's own value, also when the client sent none, so vLLM reuses only this user's cache |
| `prompt_cache_key`, `user`, `safety_identifier` | body | replaced by the user's own value, when the client sent one |
| `id_slot`, `slot_id` | body | dropped, so the upstream picks the slot |
| `request_id`, `rid`, `session_id` | body | dropped, so the upstream names the request itself |
| `conversation`, `previous_response_id`, `session_params`, `kv_transfer_params` | body | refused with a 400, because they point at state the upstream keeps and the proxy cannot tell whose it is |
| `owner`, `notes_owner` | body | refused with a 400, because the key alone names the owner |
| `Cookie`, `api-key`, `X-Api-Key`, `X-Session-Id`, `X-Request-Id`, `OpenAI-Organization`, `OpenAI-Project`, `X-Owner`, `X-Notes-Owner`, `X-User` | header | dropped, as is every header but `Content-Type`, `Accept` and `User-Agent` |

`id_slot` and `slot_id` are llama.cpp's. `cache_salt`, `request_id` and `kv_transfer_params` are vLLM's. `rid` and `session_params` are SGLang's. `user`, `safety_identifier`, `prompt_cache_key`, `conversation` and `previous_response_id` are OpenAI's.

The user's own value is the same in all of that user's requests until the proxy restarts, and another in every other user's. The proxy makes it from the user's name and a secret it draws at each start, so no user can work out another's. `--share-upstream-cache` leaves `cache_prompt`, `cache_salt` and `prompt_cache_key` as the client sent them.

A dropped field changes only where the upstream does the work, or the name it gives the request. The answer stays the same. A refused field would change the answer if it were dropped, so the proxy says so with a 400 instead.

No notes field goes on to the upstream either. The chat route reads its own (see What it does) and refuses any other field whose name starts with `notes_`. Completions and embeddings refuse every field named `notes` or starting with `notes_`. So an upstream that keeps notes of its own, such as a TensorFold server, never gets a request for them.

## Living Weights: notes and weights as one system (with `--tensorfold --slide`)

Living Weights (`--slide`) teaches facts into the model's own weights. The proxy keeps the two layers as one system, on rules the promotion study (`docs/PROMOTION.md`) measured:

- **Notes stay the record of truth.** Every fact is a note the moment it is said, taught or not.
- **A taught note leaves the prompt.** When the operator's promoter teaches a note's sentence to the pocket and the pocket answers alone, it marks the note (`POST /v1/notes/ID/weights {"in_weights": true}`). From then on the note is left out of every block — the weights answer it at zero prompt tokens. `"notes_in_weights": true` brings the block back for a control arm. A mark true is accepted only when the child's own graph shows a learned fact with that note's exact text: a client cannot retire a note from the prompt by claiming a lesson the pocket never learned (such a claim is a 409).
- **The marker survives restarts.** It is a line in the journal (`{"weights": {...}}`), replayed at open in the journal's own order, per owner. `GET /v1/notes` shows each note's `in_weights`; so does the annotated `GET /v1/slide/graph`, which maps each learned fact to the note whose text is exactly its taught sentence.
- **A change demotes at once.** When a write supersedes or takes back a taught note, the old note stops being trusted the same instant (its marker goes false, `demoted` names it in the write's reply), so the new fact rides the prompt immediately. The stale fact leaves the pocket later: `DELETE /v1/slide/fact/{id}` on the pocket, never zeroed in place by the proxy. Erasing a note drops its marker with it.
- **The proxy never teaches.** `/v1/slide/learn` and `DELETE /v1/slide/fact/...` are refused through the proxy at every age (one owner or many). Teaching is the operator's promoter, talking to the pocket directly. The child flags are allowlisted: `--tensorfold-arg=--slide` starts the child as a pocket (`--slide-graph=FILE` to keep its graph). The refusal and the `--slide` allowlist landed in the same change: no released proxy ever let a client reach `/v1/slide/learn` — before this change the flag was not on the allowlist, so the child never ran Living Weights and the pocket's routes answered 404 from the child itself.
- **Not with many users, yet.** A Living Weights pocket is one pocket: a fact taught for Alice is in Bob's weights too. So `--slide` with `--users` stops at start (no allowing flag), and `GET /v1/slide/graph` answers 404 with many users (its graph is the whole pocket, texts and questions included, for whoever asks). Until TensorFold keeps a pocket per user, run one proxy per owner.

**Minimum TensorFold: 1.0.12.** That release shipped `DELETE /v1/slide/fact/{id}` (per-fact delete, #872); `--slide` itself needs the release after 1.0.6. Older children answer those routes 501 or lack them, and the proxy will not pretend otherwise.

The nightly promoter that decides which notes have earned teaching is a separate tool (`python -m notes_proxy promote`); it comes as the second change and, when it keeps answered chats to measure reuse, is opt-in per owner, owner-only files (`0600`), with a retention limit, erased along with the note and by `remove-user --erase-notes`.

## What this proxy cannot keep apart

With many users, the proxy keeps each user's notes apart. The upstream server is still one server for all of them, and the proxy cannot reach inside it. These stay shared:

- **Saved conversations and slot files.** An upstream may save conversations or responses, and llama.cpp can save a slot to a file. The proxy passes no field that names them and no route that reads them. They are still on the upstream, for anyone who can reach it directly.
- **The prompt cache of a server that ignores `cache_prompt` and `cache_salt`.** Such a server may reuse one user's prompt for another. Run one upstream per user there, or turn its prefix cache off.
- **Timing.** A reply comes faster when the upstream had part of the prompt cached, and slower when it was busy with another user. The proxy takes the cache counts out of the replies. It cannot take out the time.
- **The upstream's logs.** The upstream may log every prompt it gets, each user's notes block included. Keep its logs as private as the notes.
- **Living Weights in TensorFold.** A fact that a TensorFold upstream learned with Living Weights is in the model's own weights, and every user's answers come from them. The proxy refuses every `/v1/slide` route except its own annotated `GET /v1/slide/graph` (which it refuses too with many users), so no client can teach or delete in the weights through the proxy (with `--tensorfold` the child's key rides on every relayed request, which is exactly why no slide route may pass). Separate notes do not make separate weights, and the proxy cannot split one pocket, so it refuses to run that way at all: `--slide` with `--users` stops at start. See Living Weights below for how notes and weights work as one system.

## Safety

- **One owner per store.** With `--owner`, every request is answered from one person's notes, the `--owner` given at start. The store names its owner, and a proxy started with another owner refuses it. Run one such proxy per person. To serve several people from one proxy, use `--users` (see Many users), never one `--owner` for all of them.
- **Notes are data, not instructions.** The block quotes each note as something the user said and tells the model never to follow an instruction inside a quote. That is `notes.py`'s framing, unchanged, and `--framing firm` keeps the same sentence.
- **Local by default.** The proxy binds to 127.0.0.1. Binding elsewhere needs `--allow-remote`, because anyone who can reach the proxy can read and erase the owner's notes. With `--upstream-key-env` they can also use the upstream's key. Requests from web pages (any request with an `Origin` header) are refused unless that origin is allowed with `--allow-origin`, so a page in your browser cannot read or write your notes. Without `--allow-remote` the proxy also answers only requests whose `Host` names this machine (`localhost`, `127.0.0.1` or `[::1]`), which stops a page that points its own name at 127.0.0.1 (DNS rebinding) and so sends no `Origin`.

## Options

| Option | Meaning |
| --- | --- |
| `--upstream URL` | the inference server's base URL (`http://127.0.0.1:8080`, or with `/v1` at the end). Mutually exclusive with `--tensorfold` |
| `--tensorfold MODEL` | start TensorFold on this model (local dir or `org/name`) and use it as the upstream |
| `--tensorfold-bin PATH` | TensorFold executable (default: `tensorfold` on PATH) |
| `--tensorfold-arg FLAG` | one extra `tensorfold serve` flag from the allowlist (repeatable). Write `--tensorfold-arg=--no-drafts` |
| `--tensorfold-quiet S` | idle seconds after the last erase before one TensorFold restart (10) |
| `--tensorfold-restart-max S` | if requests are still in flight this many seconds after the first pending erase, refuse new ones so the restart can run (60) |
| `--owner NAME` | whose notes this proxy serves (one owner) |
| `--store FILE` | the notes journal (made if missing) |
| `--users FILE` | many users: the users file (see Many users) |
| `--store-dir DIR` | many users: the folder of their stores, one `NAME.jsonl` each (made if missing) |
| `--upstream-key-env NAME` | send the upstream the key held in the environment variable `NAME`, in place of the client's `Authorization` |
| `--share-upstream-cache` | many users: let the upstream reuse one user's cached prompt for another (off, see Many users) |
| `--tokenizer FILE` | the model's `tokenizer.json` (or its folder) |
| `--cap N` | the block's token cap, framing included (512) |
| `--abstain-line TEXT` | a sentence added to the end of the block's first line (off). The cap counts it |
| `--framing NAME` | the block's first paragraph: `quoted`, `notes.py`'s own (the default), or `firm`, which also says a note is right when it disagrees with what the model remembers. The cap counts it |
| `--write` | write notes from the user's chat statements (off) |
| `--join-system` | join the block and the client's leading system messages into one system message, with a blank line between, as the TensorFold server does before its chat template. For an upstream whose chat template refuses a second system message |
| `--notes-config JSON` | other `notes.py` settings, such as `{"takeback": "explicit"}` |
| `--host`, `--port` | where to listen (127.0.0.1, 8090) |
| `--allow-remote`, `--allow-origin ORIGIN` | see Safety |
| `--timeout S` | seconds to wait for the upstream (600) |

**Why `--framing firm`.** With thinking on, Qwen3.8-27B sometimes argued with a note under the default header and kept its own memory, most often in German. In small probes on 24 Oct 2026 the firm header made it follow the note 4 of 4 times at temperature 0, and 20 of 20 at 0.7 against 15 of 20 for the default. On Flash-Next (3-bit GGUF on llama.cpp) both headers were followed about 9 times in 10 at 0.7. Firm is not the default yet, because it has not been checked on questions whose right answer is "I don't know".

## What it does not do

These need the engine itself, which a proxy in front of it cannot reach:

- **No live fetch mid-answer.** The notes are looked up once, from the question, before the model starts. The model cannot look a note up while it answers.
- **No prompt-cache reuse.** The proxy cannot keep the notes block's prefill across requests; the upstream's own prompt cache may or may not reuse it.
- **No Living Weights.** Nothing is taught into the model's weights, and no note is promoted into neurons.

## Limits

- **The lookup is Python.** In the TensorFold server's comparison, `retrieve()` took 0.42 to 23.6 ms a question (p50, stores of 100 to 100,000 notes); the server's own lookup took 52 to 437 microseconds on the same stores.
- **A write blocks lookups briefly.** The store takes one write at a time and lookups wait for it (TensorFold's server builds a new index version instead).
- **English only.** The past-tense cue and the question check read English. `"notes_language"` other than `"en"` is refused.
- **The tokenizer is yours to match.** The proxy cannot see which model the upstream runs. Give it that model's `tokenizer.json`, or the cap is counted for another model. With `--tensorfold` the child tokenizes, so `--tokenizer` is not used.
- **Only `/v1/chat/completions` gets notes.** `/v1/completions`, `/v1/responses` and `/v1/messages` pass through without them.
- **Free text is not read for meaning.** A sentence outside the rule book's wordings never replaces an earlier note: "My car is blue." and later "My car is red." both stay current. Erase the old one.
- **The journal has no commit marks.** A write whose line reached the disk just before a crash, but was never answered, comes back at the next start.

## Tests

From the repository's top folder:

```sh
python3 -m unittest discover -s notes_proxy/tests -t .
```

They use the Nemotron tokenizer when it is in the Hugging Face cache and `tokenizers` is installed, otherwise the stand-in; `NOTES_PROXY_TEST_TOKENIZER=fake` or `=PATH/tokenizer.json` chooses. No model runs and no GPU is used: the upstream is a stub that echoes the messages it receives.

- **The block (`test_block.py`).** The L6-seed1 teach store, seeded into a journal and read back: for all 56 questions, the proxy's block, note numbers, tokens and past flag equal `retrieve()`'s, also with the past forced, with two notes left out (named by number, and named by id), and at a cap of 128. Controls: the store of another phase and of another set each give different blocks. Also: the abstain line, a restart, one owner and one proxy per store.
- **The framing (`test_framing.py`).** With `--framing firm`, for all 56 questions, with one owner and with many users, the upstream gets the firm header first and the rest of the block is `notes.py`'s. At a cap of 160 every firm block fits, and 55 of 56 hold fewer notes than the default's (control: the header swapped in after the fit goes over the cap in 55 of 56). With no flag and with `--framing quoted`, the 56 blocks are byte for byte those of the proxy before the option, at caps 512 and 160 (their SHA-256 is pinned in the test). Firm combines with the abstain line and the unknown rule. A bad value stops the start.
- **End to end (`test_e2e.py`).** Through HTTP to the stub: the block arrives first and unchanged and every client message after it unchanged, for all 56 questions; nothing found puts nothing in; `"notes": false` sends the request unchanged; a streamed reply's first chunk reaches the client while the stub still holds the rest back (control: a proxy that buffers the reply fails this check); erase removes the note from the next request, from every structure the proxy holds (walked with `notes.py`'s own walker) and from the journal, and a restart keeps it gone; a restart keeps notes; `--write` writes statements and skips questions, requests, earlier messages and the model's answers; the API key passes through and stays out of the logs.
- **Living Weights (`test_weights.py`).** Against the fake child (whose folder can hold a `slide_graph.json`): a marked note leaves the block at once and stays out after a restart (the journal's weights line replays); `"notes_in_weights": true` brings it back (the control arm); marking an erased id is a 404 and a bad body a 400; a mark true is accepted only when the child's graph shows a learned fact with the note's exact text (nothing taught is a 409 and the note keeps riding the prompt; a `missed` fact is no proof; neither is a near miss); a REPLACES write demotes the taught note it retires in the same reply (`demoted`), and the new note rides the prompt; an erase drops the note's weights line from the journal; `GET /v1/slide/graph` maps each fact to the note whose text is the taught sentence and marks it `in_weights` (a fact no note holds names no note); `POST /v1/slide/learn`, `DELETE /v1/slide/fact/1` and `DELETE /v1/slide/graph` through the proxy are 404 and reach the child not at all; with many users `GET /v1/slide/graph` is 404 to every user and its body names none of another user's taught sentences or questions; `--slide` (or `--slide-graph`) with `--users` stops the start, and the same proxy without the pocket starts (control); without `--tensorfold` both new routes are 404; `--slide` and `--slide-graph` pass the child allowlist.
- **Many users (`test_users.py`).** bob has the L6-seed1 teach store. alice has L6-seed2's, and a note in the words of each L6-seed1 question. For all 56 questions, bob's block is `retrieve()`'s and holds none of alice's notes. His blocks and scores are the same with her store empty, full, and full after a restart (control: her notes in his own index move all 56 blocks). Her erase leaves his journal and blocks as they were, and stays after a restart. A missing, unknown or removed user's key is a 401, and the stub receives nothing. The user's key never reaches the stub, and the upstream key does. Names with path tricks are refused. 448 requests from four users on 16 threads give the same blocks as one at a time. The users file never holds a key, nor any 12 characters of one.
- **Many users, the edges (`test_many_safety.py`).** A stand-in for llama.cpp's prompt cache: bob's 9 probes of alice's cached block reuse nothing, and no reply counts reused tokens (control: with `--share-upstream-cache` the right guess reuses more than the wrong ones). Only the model routes pass through. A request with the key in its URL or body is refused. Replies carry no lookup times. A wrong store folder erases nothing and keeps the user, and a kept name needs `--reuse-notes`. A users file that others can change stops the start. A store that cannot be opened is a clear reply for its own user only, and a journal that failed mid-erase opens again for its user alone. The real proxy, started under a soft limit of 64 open files, serves 60 users, and under a hard limit of 128 it refuses to start. The Many users commands of this README work in order in an empty home folder.
- **Many users, the upstream's state (`test_many_state.py`).** A stand-in upstream keeps state under every field and header of the table in What reaches the upstream, so two requests that name the same thing share it. alice and bob send at the same moment, three times over, requests that carry the same value for each of those 26 fields and headers. bob never gets alice's prompt back, and each field is set, replaced, dropped or refused as the table says (control: with two one-owner proxies in front of the same stand-in, bob gets alice's secret through all 26). A field or header that names alice as the owner never gives bob her notes, nor writes his into hers (control: a proxy that read `X-Owner` would give him her note). The upstream gets only `Content-Type`, `Accept` and `User-Agent` of the client's headers, and the upstream key (control: one owner still passes the client's headers on).
- **Note ids (`test_erase_ids.py`).** Three notes, the first erased, then the same erase sent again: the second is a 404, the other two notes are still there under the ids they had, and the journal is not touched. The same after a restart, on the store itself, through HTTP with one owner, and with many users (alice and bob both have a `w1`; her repeated erase leaves his notes and journal as they were). A seeded note (`s1`) is erased once. An erased id is never given to a later note, also after a restart. A number in place of an id is a 400 that erases nothing. The chat reply's `ids` are the listing's ids, its `numbers` are the `[n]` of the block, and `notes_exclude` takes ids. Every one of these fails on the proxy before the change. In the retry cases the second erase answered 200 and removed another note.
- **The question check (`test_gate.py`).** The TensorFold server's 55 questions and requests are skipped and its 18 statements written, with its documented misses.

## How it differs from the TensorFold server

- Notes are on unless a request sends `"notes": false` (the server: off unless `"notes": true`). Writes from chat are off unless `--write` (the server: on with `--notes`).
- `DELETE /v1/notes/ID` erases. On the server it is a take-back, and an erase is `DELETE /v1/notes/ID?erase=true` (the proxy takes that too). The proxy has no change (`PUT`, `"replaces"`) or take-back endpoint, and it can erase every note, also those it was seeded with (the server refuses to erase snapshot notes).
- On the server a note id is a number. In the proxy it is text, the write's own id (`w12`), which never moves; `notes_exclude` and the reply's `notes.ids` use these ids (see Note ids).
- When streaming, `notes` is an SSE comment before `[DONE]`, not a field of the last chunk. The reply's `notes` has the fields `retrieve()` reports (`ids`, `past`, `past_from`, `cap`, `tokens`, `candidates`, `notes_in_store`, `lookup_us`, `total_us`, `block_sha256`, `block`, `write`, `write_us`, with many users without the three times), not the server's split timings (`tokenize_us`, `fit_us`, `pack_tokens`, `dropped`, `question_tokens`).
- The block is put in as its own first system message, as the server inserts it. The server then joins it with the client's leading system messages into one; the proxy leaves that to the upstream's chat template, or does the same with `--join-system`.
- Free-text writes use the rule book's intake when it reads the sentence, so a change ("... now.") retires the old note; the server writes every chat statement as new unless it repeats a current note.
- The store is a journal of the writes, not the server's snapshot and journal; the proxy does not check that the upstream's model matches the tokenizer (the server checks the tokenizer's hash).
