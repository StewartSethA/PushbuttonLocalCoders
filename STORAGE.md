# Model download storage

The three local launchers store weights in `--local-cache DIR`, or
`CLAUDE_LOCAL_CACHE`, defaulting to `~/.cache/pushbutton/llama`. Before downloading
they print the resolved model folder and exact total size from Hugging Face
metadata. Files live under `models--OWNER--REPO/REVISION/`, preserving subfolders
and pinning all GGUF shards to the same revision. llama.cpp loads the first shard
with `-m` and discovers its peers locally.

Downloads refuse to start when the remaining weights plus a reserve of
**max(10 GiB, 5% of the filesystem)** exceed free space. Completed files are
excluded; allocated aria2 partial-file blocks are credited only for parallel
resume. Unknown sizes, missing quants, or incomplete/ambiguous shard sets are
rejected before downloading. If a folder-management `validate_download_space`
function is loaded, its policy is checked too.

Quant-only selectors such as `IQ3_XXS` select the standard model, not optional
`-mtp` builds. To request an MTP build, use `IQ3_XXS-mtp` or its exact GGUF
filename; exact shard filenames still select the complete shard set.

## Parallel downloads and resume

aria2c downloads up to four files concurrently, using up to three connections
per file, preallocation, per-chunk retries, and live percentage/speed/ETA output.
Hugging Face LFS SHA-256 checksums, when available, are checked by aria2c.
Partial files and their `.aria2` control files stay in the model folder after a
failure or interruption. **Keep both files and re-run the same launcher** to
resume; a partial-download message identifies them. New upstream revisions use a
new directory rather than mixing old and new shards.

If aria2c cannot be installed or fails to download, the Hugging Face CLI is used
with live progress, one file at a time. Its private environment, when needed, is
stored under `$CLAUDE_LOCAL_STATE/download-venv` (default:
`~/.local/share/pushbutton/claude-local/download-venv`). Hugging Face authentication
uses `HF_TOKEN` or the saved Hugging Face token. The fallback uses its own resume
metadata and may restart an aria2 partial file, but skips completed shards;
space is checked again to allow
for temporary copies. `--no-parallel` disables aria2c and automatic installation.

Successful downloads append JSON records to `$CACHE_DIR/.download-log`, including
UTC timestamp, model spec, total bytes, pending-file bytes, elapsed seconds,
download method, and average logical bytes per second. For resumes this is a
file-level throughput estimate, not a measurement of bytes transferred over the
network. Cached-only launches do not add download events.

Existing content-addressed llama.cpp/Hugging Face blobs in the chosen cache are
reused with hard links when metadata identifies their SHA-256 and size; they are
never copied or deleted. Other cache layouts are left untouched and may require
a new download into the revision directories.
Choose a larger disk with `--local-cache /mnt/bigdisk/models`, select a smaller
quant, or free space if validation refuses a download. Offline metadata queries,
unavailable packages, authentication failures, and unreachable servers can
still prevent downloading; the launcher reports the error and does not start
the backend.
