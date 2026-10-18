# How this repository relates to TensorFold releases

The intended route is to contribute the reviewed Notes implementation and its host integration upstream. Once Ash merges it and includes it in a TensorFold release, users of that release get Notes through TensorFold's normal build and explicit feature switches. That release does not need to download this repository at runtime.

This repository remains the ThinkOff team's home for module source, documentation, examples, reproducibility and development. An upstream contribution and this separate home can coexist. Acceptance and release timing belong to TensorFold's maintainer.

## What works today

The current package is a source snapshot, not an independently installable add-on. `SOURCE.json` pins the tested **ThinkOffApp/TensorFold fork commit**, and `tools/stage_tensorfold.py` checks out that complete host. That host supplies the surrounding server, request routing, model tokenizer, prompt handling and build integration. The tool verifies that its 42 module files equal this repository's export.

The export alone does not add all that integration to an arbitrary TensorFold release. Applying just these files on top of stock TensorFold is not a supported installation procedure. The pinned host setup is the reproducible path until a separately tested release integration is available.

## Release process

1. Finish the production gates in the README, including erase, write latency and Metal/CUDA streaming/cache tests.
2. Submit the module **and required host integration** as reviewable TensorFold PRs, with exact source and test receipts.
3. For each supported pairing, record the module tag, full TensorFold commit or release tag, source hashes, platform checks and known limitations. Re-run the checks when either side changes.
4. After upstream acceptance, update this repository's manifest and setup instructions to a tested upstream commit or release. Until then, identify the tested fork explicitly.

There is no automatic synchronization between the repositories today. A maintainer updates the pinned source and verifies it; a new commit here does not silently change a TensorFold release. Maintaining an independent fork or patch set is possible, but each supported release would need its own tested integration.

As of 18 October 2026, TensorFold's latest release is v1.1.0. In [the Notes discussion](https://github.com/ashhart/TensorFold/issues/1012), Ash targets a later 1.1.x release once its PRs pass the stated gates. This package does not establish that Notes has shipped in an official release.
