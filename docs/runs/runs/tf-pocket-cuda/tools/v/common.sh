# Shared guard for the verify scripts: the GPU lock, and no other engine, test, nsys or dedup process.
H=~/tf-pocket-cuda
declare -A M=([A]=~/.cache/huggingface/hub/models--TensorFold--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/snapshots/d9d758fb83953437f7263256b0d96157e2a348b8 [T]=~/lw-runs/gb10-a-L7-seed1/models/L7-seed1.main [E]=$H/models/A-empty64)
declare -A CLI=([before]=$H/before/zig-out/bin/tensorfold [after]=$H/src/zig-out/bin/tensorfold)
declare -A CLISHA=([before]=0f0fe4b069fe822733e908fa27f72099f35ed5efbaf06dc4fe0c3154802e80d7 [after]=37efd269fa6de828d7e82dc7a69a433c3b3eaed59258e2e48f8ee6f91cd9535f)
declare -A TEST=([before]=$H/before/zig-out/bin/tf-cuda-test [after]=$H/src/zig-out/bin/tf-cuda-test)
take_lock() { exec 9>$H/GPU.lock; flock -n 9 || { echo "$(date -u +%T) refusing: GPU.lock held"; exit 3; }; }
busy() { pgrep -x tensorfold-nati >/dev/null || pgrep -x tensorfold >/dev/null || pgrep -x tf-cuda-test >/dev/null || pgrep -f "^[^ ]*nsys profile" >/dev/null || pgrep -f "^[^ ]*python[0-9.]* [^ ]*dedup_identical_copies" >/dev/null; }
wait_free() { while busy; do echo "$(date -u +%T) waiting: another GPU or dedup process: $(pgrep -a -x tensorfold-nati; pgrep -a -x tensorfold; pgrep -a -x tf-cuda-test; pgrep -af dedup_identical_copies | grep -v pgrep)"; sleep 30; done; }
check_clis() { for b in before after; do s=$(sha256sum < ${CLI[$b]} | cut -c1-64); [ "$s" = "${CLISHA[$b]}" ] || { echo "refusing: ${CLI[$b]} sha256 $s"; exit 3; }; done; }
hashes() { for m in A T E; do (cd ${M[$m]} && for f in config.json *.safetensors; do echo "$m $f $(sha256sum < "$f" | cut -c1-64)"; done); done; }
smi() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu --format=csv,noheader; }
stop_check() { if [ -e $H/raw/v/STOP ]; then echo "$(date -u +%T) STOP file present: stopping"; exit 5; fi; }
