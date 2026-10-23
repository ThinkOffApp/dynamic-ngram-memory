# Notes proxy

**TensorFold Notes for any OpenAI-compatible inference server.**

The notes proxy is a small server that sits between your chat client and the server that runs the model: llama.cpp's `llama-server`, vLLM, Ollama, LM Studio, SGLang or TensorFold itself. When a chat request passes through it, the proxy looks up what this user has told it before (their notes), puts the notes that fit the question into the request as the first system message, and forwards the request. The model then answers with the notes in front of it. Nothing about the model or its server changes.

The lookup is not a copy. The proxy calls the test suite's own note memory, [`tests/zengine/notes.py`](../tests/zengine/notes.py): the same n-gram matching, scoring, hop, 512-token cap and framing that the paper measures and that the TensorFold server's lookup was checked against. For every question the proxy sends exactly the block `NoteStore.retrieve()` returns, byte for byte (tested below).

## Run it

You need Python 3.10 or later and the `tokenizers` package (`pip install tokenizers`), and the model's `tokenizer.json`. The 512-token cap is counted with the model's own tokenizer, as in the paper. Run the commands from this repository's top folder.

In front of llama.cpp (`llama-server -m model.gguf --port 8080` running). A GGUF file has no `tokenizer.json`: use the one from the model's original Hugging Face repository.

```sh
python3 -m notes_proxy --upstream http://127.0.0.1:8080 --owner me --store ~/notes/me.jsonl --tokenizer ~/models/my-model/tokenizer.json
```

In front of vLLM (`vllm serve MODEL --port 8000` running):

```sh
python3 -m notes_proxy --upstream http://127.0.0.1:8000 --owner me --store ~/notes/me.jsonl --tokenizer ~/.cache/huggingface/hub/models--ORG--MODEL/snapshots/HASH/tokenizer.json
```

Then point your chat client at `http://127.0.0.1:8090/v1` instead of the server's own address. Add a note and ask about it:

```sh
curl -s http://127.0.0.1:8090/v1/notes -H "Content-Type: application/json" -d '{"text": "My sister Anna drives a red Volvo."}'
```

```sh
curl -s http://127.0.0.1:8090/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "m", "messages": [{"role": "user", "content": "What car does Anna drive?"}]}'
```

The reply is the server's own, with one more field, `notes`: the ids of the notes given, their tokens, and whether the question was read as one about the past.

To try it on a test set's notes, make a store from one of the generated sets first:

```sh
python3 -m notes_proxy seed --set tests/examples/L6-seed1.json --phase teach --owner user --store /tmp/l6.jsonl
```

## What it does

- **`POST /v1/chat/completions`.** The last user message is the question (a string, or its text parts joined, as the TensorFold server reads it). If the lookup finds notes, the block goes in first as `{"role": "system", "content": block}`; the client's own messages follow it unchanged. If it finds nothing, nothing is put in. Then the request goes to the upstream server.
- **Streaming.** A streamed reply is passed on line by line as the upstream sends it, never held back. The `notes` object comes as an SSE comment line, `: notes {...}`, just before `data: [DONE]`. Every SSE parser skips comment lines, so no client meets a chunk it does not expect. The same ids, tokens and past flag are also in the reply headers `X-Notes-Ids`, `X-Notes-Tokens` and `X-Notes-Past`. A reply that is not streamed gets the `notes` field added at the end of its JSON; the upstream's bytes before it are left as they were.
- **Request fields**, as the TensorFold server reads them: `"notes": false` sends the request on without notes; `"notes_past": true/false` forces the view of the past (absent: the question decides); `"notes_exclude": [ids]` leaves those notes out; `"notes_cap": n` sets the cap; `"notes_echo": true` puts the block itself in the reply; `"notes_write": false` keeps this message out of the notes. A field of the wrong type is a 400. The proxy removes these fields before forwarding, so the upstream never sees them.
- **Every other route** (`/v1/models`, `/v1/completions`, `/v1/embeddings` and the rest) is passed through as it is.
- **The upstream's API key**, if it needs one, is the client's: the `Authorization` header is passed on as sent. The proxy never stores it and never logs it. It logs one line per request: the route, the status, the note ids and times, never a message or a note's text.

## Notes API

| Request | What it does |
| --- | --- |
| `GET /v1/notes` | the current notes; `?history=1` also lists the replaced and taken-back ones |
| `GET /v1/notes/ID` | one note |
| `POST /v1/notes` `{"text", "subject"?}` | a new note, through the intake check (201 when written, 200 when it was already known) |
| `DELETE /v1/notes/ID` | erase: the note is removed from the store, its journal and every cache the proxy holds, as if it had never been said |

An erase also removes the writes that were kept out as the same as that note, and the take-backs of it. Ids after the erased note move down by one; the reply lists every move. The erase reply always says `"complete": false`: the proxy cannot reach the upstream server's prompt cache or logs, and Python does not overwrite memory it frees.

The notes are kept in the `--store` file, a journal with one write per line. Each write is on disk (synced) before it is answered or used. At startup the journal is replayed, so a restart keeps every note with the same ids. An erase writes the journal anew without the erased writes. The file holds the notes in plain text: keep it where only you can read it (the proxy makes it readable by its owner only).

## Writing notes from chat (`--write`, off by default)

With `--write`, the proxy writes notes from what the user says, by the TensorFold server's rule:

- only the conversation's last message, and only when it is the user's own; never a system message, an earlier message or the model's answer (an answer can be wrong, and storing it would keep the mistake);
- only a statement: a message that reads as a question ("Where do I live?", "what car does Anna drive") or a request ("Tell me...", "Remind me...") is skipped, as is anything over one line or 400 bytes. The question check is the server's own (English only), ported to Python;
- through the intake check, after the lookup, before the request is forwarded: the next request finds the note.

A sentence in the wordings the rule book reads ("My favourite colour is green.", "I live in Oslo now.") goes through the rule book's intake check (`intake.classify`): new, replaces an earlier note, already known, or implied. Any other sentence is written as the TensorFold server writes every chat statement: already known if a current note of the same subject says the same words, otherwise new. The reply's `notes.write` says what happened, or why the message was skipped.

## Safety

- **One owner per proxy.** Every request is answered from one person's notes, the `--owner` given at start. The store names its owner, and a proxy started with another owner refuses it. Run one proxy per person; never put one proxy in front of several users.
- **Notes are data, not instructions.** The block quotes each note as something the user said and tells the model never to follow an instruction inside a quote (`notes.py`'s framing, unchanged).
- **Local by default.** The proxy binds to 127.0.0.1. Binding elsewhere needs `--allow-remote`, because anyone who can reach the proxy can read and erase the owner's notes. Requests from web pages (any request with an `Origin` header) are refused unless that origin is allowed with `--allow-origin`, so a page in your browser cannot read or write your notes. Without `--allow-remote` the proxy also answers only requests whose `Host` names this machine (`localhost`, `127.0.0.1` or `[::1]`), which stops a page that points its own name at 127.0.0.1 (DNS rebinding) and so sends no `Origin`.

## Options

| Option | Meaning |
| --- | --- |
| `--upstream URL` | the inference server's base URL (`http://127.0.0.1:8080`, or with `/v1` at the end) |
| `--owner NAME` | whose notes this proxy serves |
| `--store FILE` | the notes journal (made if missing) |
| `--tokenizer FILE` | the model's `tokenizer.json` (or its folder) |
| `--cap N` | the block's token cap, framing included (512) |
| `--abstain-line TEXT` | a sentence added to the end of the block's first line (off). The cap counts it |
| `--write` | write notes from the user's chat statements (off) |
| `--join-system` | join the block and the client's leading system messages into one system message, with a blank line between, as the TensorFold server does before its chat template. For an upstream whose chat template refuses a second system message |
| `--notes-config JSON` | other `notes.py` settings, such as `{"takeback": "explicit"}` |
| `--host`, `--port` | where to listen (127.0.0.1, 8090) |
| `--allow-remote`, `--allow-origin ORIGIN` | see Safety |
| `--timeout S` | seconds to wait for the upstream (600) |

## What it does not do

These need the engine itself, which a proxy in front of it cannot reach:

- **No live fetch mid-answer.** The notes are looked up once, from the question, before the model starts. The model cannot look a note up while it answers.
- **No prompt-cache reuse.** The proxy cannot keep the notes block's prefill across requests; the upstream's own prompt cache may or may not reuse it.
- **No Living Weights.** Nothing is taught into the model's weights, and no note is promoted into neurons.

## Limits

- **The lookup is Python.** In the TensorFold server's comparison, `retrieve()` took 0.42 to 23.6 ms a question (p50, stores of 100 to 100,000 notes); the server's own lookup took 52 to 437 microseconds on the same stores.
- **A write blocks lookups briefly.** The store takes one write at a time and lookups wait for it (TensorFold's server builds a new index version instead).
- **English only.** The past-tense cue and the question check read English. `"notes_language"` other than `"en"` is refused.
- **The tokenizer is yours to match.** The proxy cannot see which model the upstream runs. Give it that model's `tokenizer.json`, or the cap is counted for another model.
- **Only `/v1/chat/completions` gets notes.** `/v1/completions`, `/v1/responses` and `/v1/messages` pass through without them.
- **Free text is not read for meaning.** A sentence outside the rule book's wordings never replaces an earlier note: "My car is blue." and later "My car is red." both stay current. Erase the old one.
- **The journal has no commit marks.** A write whose line reached the disk just before a crash, but was never answered, comes back at the next start.

## Tests

From the repository's top folder:

```sh
python3 -m unittest discover -s notes_proxy/tests -t .
```

They use the Nemotron tokenizer when it is in the Hugging Face cache and `tokenizers` is installed, otherwise the stand-in; `NOTES_PROXY_TEST_TOKENIZER=fake` or `=PATH/tokenizer.json` chooses. No model runs and no GPU is used: the upstream is a stub that echoes the messages it receives.

- **The block (`test_block.py`).** The L6-seed1 teach store, seeded into a journal and read back: for all 56 questions, the proxy's block, note ids, tokens and past flag equal `retrieve()`'s, also with the past forced, with two notes left out, and at a cap of 128. Controls: the store of another phase and of another set each give different blocks. Also: the abstain line, a restart, one owner and one proxy per store.
- **End to end (`test_e2e.py`).** Through HTTP to the stub: the block arrives first and unchanged and every client message after it unchanged, for all 56 questions; nothing found puts nothing in; `"notes": false` sends the request unchanged; a streamed reply's first chunk reaches the client while the stub still holds the rest back (control: a proxy that buffers the reply fails this check); erase removes the note from the next request, from every structure the proxy holds (walked with `notes.py`'s own walker) and from the journal, and a restart keeps it gone; a restart keeps notes; `--write` writes statements and skips questions, requests, earlier messages and the model's answers; the API key passes through and stays out of the logs.
- **The question check (`test_gate.py`).** The TensorFold server's 55 questions and requests are skipped and its 18 statements written, with its documented misses.

## How it differs from the TensorFold server

- Notes are on unless a request sends `"notes": false` (the server: off unless `"notes": true`). Writes from chat are off unless `--write` (the server: on with `--notes`).
- `DELETE /v1/notes/ID` erases. On the server it is a take-back, and an erase is `DELETE /v1/notes/ID?erase=true` (the proxy takes that too). The proxy has no change (`PUT`, `"replaces"`) or take-back endpoint, and it can erase every note, also those it was seeded with (the server refuses to erase snapshot notes).
- When streaming, `notes` is an SSE comment before `[DONE]`, not a field of the last chunk. The reply's `notes` has the fields `retrieve()` reports (`ids`, `past`, `past_from`, `cap`, `tokens`, `candidates`, `notes_in_store`, `lookup_us`, `total_us`, `block_sha256`, `block`, `write`, `write_us`), not the server's split timings (`tokenize_us`, `fit_us`, `pack_tokens`, `dropped`, `question_tokens`).
- The block is put in as its own first system message, as the server inserts it. The server then joins it with the client's leading system messages into one; the proxy leaves that to the upstream's chat template, or does the same with `--join-system`.
- Free-text writes use the rule book's intake when it reads the sentence, so a change ("... now.") retires the old note; the server writes every chat statement as new unless it repeats a current note.
- The store is a journal of the writes, not the server's snapshot and journal; the proxy does not check that the upstream's model matches the tokenizer (the server checks the tokenizer's hash).
