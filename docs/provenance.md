# Source provenance and evidence

The exported files are byte-for-byte Git blobs from ThinkOffApp/TensorFold commit `2c9e759964fd2af1b47d79fde04b77ee564f2e54`, the stable Notes review snapshot. `SOURCE.json` records every file and SHA-256. The staging tool verifies both the export and the pinned host before running tests. No model checkpoints, working Git history, credentials or private run logs are included. The paper material added later (`docs/receipts/`, `docs/notes-raw/`, `docs/rejudge/` and `tests/`) includes run logs and results with home paths, machine names and handles removed; see [PAPER-RESULTS-DETAIL.md](PAPER-RESULTS-DETAIL.md).

The module depends on surrounding TensorFold server integration. Fetching that exact host also supplies the request parser, model tokenizer, server routing, engine interface and build definitions that the module imports. A future standalone library needs explicit interfaces for these dependencies.

Historical experiment reports remain in the source repository at the pinned revision:

- [Notes integration and equivalence report](https://github.com/ThinkOffApp/TensorFold/blob/2c9e759964fd2af1b47d79fde04b77ee564f2e54/NOTES-IN-ENGINE.md)
- [Live decode design and experiments](https://github.com/ThinkOffApp/TensorFold/blob/2c9e759964fd2af1b47d79fde04b77ee564f2e54/NOTES-DECODE.md)
- [Write scaling limitations](https://github.com/ThinkOffApp/TensorFold/blob/2c9e759964fd2af1b47d79fde04b77ee564f2e54/NOTES-WRITE-SCALING.md)

Those reports describe their own machines, revisions and limitations. They are historical evidence, not a claim that the production gates in this repository have passed. Experimental uncertainty, bounded-write and promotion branches are excluded from this initial snapshot pending their separate reviews.
