# Package validation — 18 October 2026

Package source: `2c9e759964fd2af1b47d79fde04b77ee564f2e54` in ThinkOffApp/TensorFold. This validates the stable export, not the experimental branches.

| Check | Observed result |
| --- | --- |
| Exact source export | All 42 files match their Git blob SHA-256 values in SOURCE.json |
| Fresh host staged from local repository | Same pinned commit and all 42 module hashes verified |
| Fresh host fetched from the public source URL | Same pinned commit and all 42 module hashes verified |
| CPU server tests, Zig 0.17.0, native macOS | 319/319 pass; all 3 build steps pass; test execution reported 10 minutes, peak RSS 7 GiB |
| Tampered module export | Refused before creating a checkout |
| Existing destination | Refused; existing checkout preserved |
| Tracked file selection | Host work directory, model files and working Git history excluded |
| Text scan for selected private patterns | No home-directory paths, private host IP prefix, API-key prefix or private-key header matched |

The CPU test invocation is `zig build test-server-cpu --summary all`, launched by the staging tool's `--check` option after verifying compiler version and hashes. The local-source checkout was used for that run; the public fetch was verified separately. The pattern scan is a limited check, not a comprehensive security audit.

No new model training, GPU inference benchmark, one-million-note load qualification or end-to-end erase guarantee is established by these package checks. The README's production gates remain open. The repository is initially private staging for review.
