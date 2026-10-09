# Storage configuration

## Folder hierarchy

`PUSHBUTTON_DIR` is the install root (default `~/.local/share/pushbutton`);
the repository is installed in its `PushbuttonLocalCoders` child. Direct
repository launches report the actual code directory.

* `CLAUDE_LOCAL_STATE`: `~/.local/share/pushbutton/claude-local`. Contains
  `logs/`, CUDA toolkits, micromamba, llama.cpp sources/builds, `templates/`,
  generated plans/gateway configuration, startup lock, MCP configuration, and
  Qwen frontend configuration under `frontends/`.
* `CLAUDE_LOCAL_CACHE`: `~/.cache/pushbutton/llama`. Contains GGUF weights,
  repository metadata and cache blobs. Usually the largest directory.
* `PUSHBUTTON_CONFIG_DIR`: `~/.config/pushbutton-local`. Contains `folders.json`
  and `placement.json`. `PUSHBUTTON_PLACEMENT_CONFIG` or `--placement-config`
  can select a different placement file.

Upstream frontend installers also manage their own executables and user data
(for example `~/.claude`, `~/.hermes`, and `~/.local/bin`). Folder choices
do not relocate those upstream installations.

## Persistent schema and precedence

`$PUSHBUTTON_CONFIG_DIR/folders.json`:

```json
{
  "schema_version": 1,
  "folders": {
    "state_dir": "/home/user/.local/share/pushbutton/claude-local",
    "cache_dir": "/mnt/nvme/models",
    "config_dir": "/home/user/.config/pushbutton-local"
  },
  "created_at": "2026-10-02T00:00:00+00:00"
}
```

Paths are made absolute and symlinks resolved for overlap checks. State, cache,
and config must be distinct and may not contain one another. Missing directories
are created; unwritable directories stop startup with a path-specific diagnostic.
The code root may contain the state directory, as in the default layout.

Environment variables override saved values; `--local-cache` has highest cache
precedence. Choices made during installation/first startup and environment
values used at startup are persisted in `folders.json`.
The config directory variable selects where to look for the file. If moving that
directory manually, also set `PUSHBUTTON_CONFIG_DIR` in your shell/service
environment. Installed Claude/coder wrappers remember the chosen config location
unless the environment overrides it. Corrupt or unsupported configuration warns
and falls back to defaults, never executes content from JSON.

When a direct first startup chooses a custom config directory without an
explicit `PUSHBUTTON_CONFIG_DIR`, a small schema-valid `folders.json` locator
remains at the previous/default lookup location. Later shells follow it to the
authoritative custom file, including subsequent edits. An explicit config
environment override does not create this default locator. Broken or cyclic
locators warn rather than silently replacing the damaged files.

Interactive installation and first startup show usage and offer customization;
press Enter to retain each displayed path. Non-TTY operation never prompts.
Use `--quiet` to suppress the routine startup folder banner. `--folders` and
`--system-info` explicitly print paths, free space, usage and environment values
without requiring GPU drivers or starting a server.

## Download policy

The shared `lib/pushbutton_folders.sh` helper resolves exact GGUF quant files
from Hugging Face metadata, including split weights. It subtracts complete
cached files, not incomplete temporary downloads, and requires remaining bytes
plus **10 GiB** free on the cache filesystem. Multi-model plans are checked
before provisioning, with a fresh check before each download. The resumable
downloader additionally reserves 5% of the filesystem when that exceeds 10 GiB.
Both checks must pass before weights are downloaded and llama.cpp loads them.

For offline checks, store the Hugging Face model API response at
`<cache>/models--OWNER--REPO/metadata.json` (online checks also cache this response).
Set `HF_HUB_OFFLINE=1` or `PUSHBUTTON_HF_OFFLINE=1` to use it without a network
request. It must contain `siblings` entries
with `rfilename` and exact byte `size` (or `lfs.size`). Include the blob digest
(`lfs.sha256`) to recognize cached blobs; the optional repository `sha`
identifies snapshot paths.
Only use metadata corresponding to the actual cached revision. For example:

```json
{
  "siblings": [
    {"rfilename": "model-Q4_K_M.gguf", "size": 123456789}
  ]
}
```

All parts of a split GGUF must be listed. This metadata is also a fallback when
the API is unavailable, not permission to bypass the free-space check.

Insufficient space or unavailable/ambiguous size metadata fails closed with
exit status **2**. The diagnostic names the cache disk/mount and suggests
`--local-cache /bigger/disk`, a smaller quant, or freeing space. No disk failure
enables unified-memory spill. The planner's VRAM envelope is an estimate, not a
substitute for exact file metadata. These are preflight checks, not a reservation:
other processes can still consume disk space after validation.

## Migration to another disk

1. Stop every launcher/backend using the old directories.
2. Check the destination with `df -h /mnt/nvme`; create a private/group-owned
   destination as appropriate.
3. Preserve the whole cache layout and symlinks:

   ```bash
   mkdir -p /mnt/nvme/pushbutton-models
   cp -a "$HOME/.cache/pushbutton/llama/." /mnt/nvme/pushbutton-models/
   export CLAUDE_LOCAL_CACHE=/mnt/nvme/pushbutton-models
   claude-local --folders
   ```

4. Update `cache_dir` in `folders.json` (or persist the environment override in
   your shell/service). Move state/config similarly, but never nest the folders.
5. Successfully launch using the new paths before deleting the old copy.

A two-GPU workstation can keep builds/logs on the system SSD and weights on NVMe.
Shared-cache users should retain separate state/config folders and share only a
trusted group-owned cache. Coordinate downloads externally across users/hosts;
their private startup locks do not provide cross-user disk reservations.

## Troubleshooting

* **Permission/creation failure:** inspect the named directory and its parent,
  ownership, write/search permissions, read-only mount status, quotas and `df -h`.
  Choose a writable path rather than running the launcher as root.
* **Overlap/environment conflict:** unset stale overrides or choose disjoint
  state/cache/config paths. Symlink aliases also count as overlapping.
* **Corrupt config:** repair valid JSON/schema version 1 or move the damaged
  file aside and run again to configure defaults.
* **Unknown download size:** check connectivity/access to Hugging Face and
  repository/quant spelling. Startup refuses rather than guessing disk usage.
* **Low/full disk:** choose a bigger cache mount or a smaller quant. After
  stopping backends, inspect stale `.downloadInProgress` files; incomplete files
  are not counted as finished weights. Do not delete files used by other users.
* **Disk fills during download:** preflight cannot stop another process filling
  the disk. Follow the printed log command, check `df`, stop competing writes,
  and retry only after creating room.

Space queries support Linux and macOS; the NVIDIA provisioning/launch paths
still have their existing Linux tool requirements.
## Model download storage

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
