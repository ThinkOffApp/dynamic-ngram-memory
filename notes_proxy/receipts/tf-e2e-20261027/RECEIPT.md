# Notes Proxy TensorFold live e2e, 27 Oct 2026

Branch `notes-proxy-tf-backend` 913d571 plus child-start timeout = `--timeout`.
Binary: `/tmp/tf-ashhart-main/zig-out/native/bin/tensorfold-native` 1.0.0 from ashhart main 6eaa4cd5.
`:8091` Nemotron was not touched and stayed healthy.

Raw JSON: this folder, and `notes_proxy/receipts/tf-e2e-20261027/` on the branch.

Question used: `What car did my sister Anna drive yesterday?`
Note: `My sister Anna drives a red Volvo.`

## 2B (mlx-community/Qwen3.5-2B-MLX-4bit)

JSON in this folder (`00-health.json` … `07-health-after.json`, `RECEIPT.json`).

| check | result |
| --- | --- |
| 1. After erase, same question has `notes.ids []` and the reply has no Volvo | pass. chat1: "My sister Anna drove a red Volvo yesterday." chat3: does not know Anna. pids 98380 → 99155. |
| 2. Child pid replaced | pass. 98380 → 99155 |
| 3. Second prompt shows `usage.prompt_tokens_details.cached_tokens` > 0, then ~0 after restart | fail. cached_tokens 0, 0, 0. TensorFold docs: Qwen3.5-2B keeps no prompt states. |

## 30B (Vontra/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit)

JSON in `30b/`. Context 4096, `--no-thinking`, own loopback port. `:8091` still 18.7 GB active.

| check | result |
| --- | --- |
| 1. After erase, `notes.ids []` and no Volvo | pass. chat1: "My sister Anna drives a red Volvo. (this is the current conversation)". chat3: "I don't have access to your personal information…". |
| 2. Child pid replaced | pass. 99843 → 1381. erase-status pending → restarting → complete (restart_seq 1). |
| 3. cached_tokens on the second identical prompt, then ~0 after restart | fail. cached_tokens 0 on chat1, chat2, and chat3. Direct child calls (store true, store false, omit store) also report cached_tokens 0. Second prefill is a bit faster (0.042 s → 0.032 s) but usage does not show a cache hit. |

Check 3 is the one that can fail, and it failed on both models: this native 1.0.0 server did not put prefix hits in `cached_tokens` for these requests.
