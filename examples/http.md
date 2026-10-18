# Notes HTTP examples

These requests require a running TensorFold server started with `--notes SNAPSHOT`. Add its configured authorization header where needed. The examples use fictitious facts and a local server. Replace `ID` with the returned integer note ID.

Add a fact:

```sh
curl http://127.0.0.1:8000/v1/notes -H 'Content-Type: application/json' \
  -d '{"text":"Mira lives in Turku.","subject":"Mira"}'
```

Change it:

```sh
curl -X PUT http://127.0.0.1:8000/v1/notes/ID -H 'Content-Type: application/json' \
  -d '{"text":"Mira now lives in Oulu."}'
```

Ask with Notes enabled, and inspect the retrieved block:

```sh
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Where does Mira live?"}],"notes":true,"notes_write":false,"notes_echo":true,"temperature":0,"max_tokens":64}'
```

List current notes with `GET /v1/notes`, or include history with `GET /v1/notes?history=true`.

`DELETE /v1/notes/ID` takes a note back and preserves its history. `DELETE /v1/notes/ID?erase=true` performs the note-store erase operation. These are distinct operations. Erase propagation beyond the note store remains under review; see the production gates in the README.

For chat requests, `notes_write:false` prevents automatic intake of the last user's statement. Starting the server with `--no-notes-write` disables writes globally. The base version's Notes fields apply to `/v1/chat/completions`, not every supported chat API.
