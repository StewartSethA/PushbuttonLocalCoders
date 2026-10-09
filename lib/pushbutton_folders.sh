#!/usr/bin/env bash
# Source this file; Python's standard library handles paths and untrusted JSON.
# get_free_space and estimate_model_size return integer bytes.
# validate_cache_space accepts required bytes; human summaries display GiB.
# Direct setup that relocates Config writes a schema-valid folders.json locator
# at the prior/default lookup location. Its config_locator=true marker follows
# folders.config_dir to the real file, so fresh shells see subsequent edits.
# Explicit PUSHBUTTON_CONFIG_DIR overrides do not create a default locator.
# Offline HF metadata: <cache>/models--OWNER--REPO/metadata.json, using the
# /api/models/OWNER/REPO?blobs=true response schema (siblings, size, lfs.sha256).
# Minimal fixture (positive sizes are bytes; include every split GGUF shard):
# {"sha":"<40-character commit hash>","siblings":[
#   {"rfilename":"model-Q4_K_M.gguf","size":123456,
#    "lfs":{"size":123456,"sha256":"<64-character SHA256 digest>"}}]}
# Unknown sizes and incomplete/ambiguous shard inventories fail closed.
# Digest/commit fields are optional for sizing, but needed to credit cached
# blobs/snapshots. No .downloadInProgress or .incomplete file is ever credited.
# Also searched: <cache>/{hub,hf/hub,llama}/models--OWNER--REPO/metadata.json.
# Successful API responses are cached atomically at the first path above.
# HF_HUB_OFFLINE=1 (or PUSHBUTTON_HF_OFFLINE=1) uses local metadata immediately.
# Offline manifests describe a known snapshot, not necessarily today's mutable
# repo main. Online checks refresh metadata to avoid stale size undercounts;
# with a valid local fallback, their network timeout is shortened to 2 seconds.
# Use validate_plan_download_space PLAN_FILE [CACHE_DIR] once before starting
# any downloads; separate per-model checks cannot guarantee aggregate capacity.
# Per-model guard signatures: validate_download_space HF_SPEC, or
# validate_download_space CACHE_DIR HF_SPEC [QUANT] for launcher integration.
# QUANT supplies the suffix when HF_SPEC is an owner/repo without :QUANT.
# All public validate_* functions return 2 for validation failures.

_PUSHBUTTON_FOLDER_LIB="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

_pushbutton_folder_python() {
    python3 - "$@" <<'PY'
import datetime
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid

GIB = 1024 ** 3
KEYS = ("state_dir", "cache_dir", "config_dir")
ENVS = ("CLAUDE_LOCAL_STATE", "CLAUDE_LOCAL_CACHE", "PUSHBUTTON_CONFIG_DIR")
DEFAULTS = ("~/.local/share/pushbutton/claude-local",
            "~/.cache/pushbutton/llama", "~/.config/pushbutton-local")
METADATA_MEMO = {}


def canonical(value):
    if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
        raise ValueError("folder paths must be nonempty strings without control characters")
    expanded = os.path.expanduser(value)
    if expanded.startswith("~"):
        raise ValueError("cannot expand home directory: " + value)
    return os.path.realpath(os.path.abspath(expanded))


def distinct(paths):
    for i, first in enumerate(paths):
        for j in range(i + 1, len(paths)):
            second = paths[j]
            if os.path.commonpath((first, second)) in (first, second):
                raise ValueError(
                    f"overlapping {KEYS[i]} ({first}) and {KEYS[j]} ({second}). "
                    f"Choose separate directories; unset {ENVS[i]} / {ENVS[j]} "
                    "or correct their environment overrides (and --local-cache).")


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("expected schema_version 1")
    folders = data.get("folders")
    if not isinstance(folders, dict):
        raise ValueError("missing folders object")
    paths = [canonical(folders[k]) for k in KEYS]
    distinct(paths)
    if not isinstance(data.get("created_at"), str) or not data["created_at"]:
        raise ValueError("missing created_at")
    return data, paths


def follow_config(config_home):
    seen = set()
    for _ in range(16):
        if config_home in seen:
            raise ValueError("cyclic folder config locator: " + config_home)
        seen.add(config_home)
        path = pathlib.Path(config_home) / "folders.json"
        try:
            data, paths = read_config(path)
        except FileNotFoundError:
            if len(seen) > 1:
                raise ValueError("folder config locator target is missing: " + str(path))
            raise
        if data.get("config_locator") is not True:
            return data, paths, config_home
        if paths[2] == config_home:
            raise ValueError("folder config locator points to itself: " + str(path))
        config_home = paths[2]
    raise ValueError("too many folder config locator hops")


def persist_config(dest, data):
    staging = dest.parent / (".folders-" + uuid.uuid4().hex + ".json")
    try:
        fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, dest)
    finally:
        if staging.exists():
            staging.unlink()


def existing_ancestor(path):
    path = pathlib.Path(canonical(path))
    while not path.exists():
        if path == path.parent:
            raise OSError("no existing ancestor for " + str(path))
        path = path.parent
    return path


def free_bytes(path):
    ancestor = str(existing_ancestor(path))
    for cmd in (["stat", "-f", "-c", "%a %S", "--", ancestor],
                ["stat", "-f", "%a %S", ancestor]):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            fields = result.stdout.strip().split()
            if len(fields) == 2 and all(x.isdigit() for x in fields):
                return int(fields[0]) * int(fields[1])
        except (OSError, subprocess.CalledProcessError):
            pass
    result = subprocess.run(["df", "-Pk", ancestor], capture_output=True,
                            text=True, check=True)
    fields = result.stdout.splitlines()[-1].split()
    return int(fields[3]) * 1024


def mount_description(path):
    try:
        result = subprocess.run(["df", "-P", str(existing_ancestor(path))],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "Filesystem/mount unavailable"


def check_directory(path):
    target = pathlib.Path(path)
    ancestor = existing_ancestor(path)
    # os.access alone reports writable under root, even for chmod 0555.
    for parent in (ancestor, *ancestor.parents):
        if not parent.is_dir() or not parent.stat().st_mode & 0o111:
            raise PermissionError("directory is not searchable: " + str(parent))
    mode = ancestor.stat().st_mode
    if not stat.S_ISDIR(mode) or not mode & 0o222 or not os.access(ancestor, os.W_OK | os.X_OK):
        raise PermissionError("directory is not writable: " + str(ancestor))
    target.mkdir(parents=True, exist_ok=True)
    mode = target.stat().st_mode
    if not mode & 0o222 or not mode & 0o111 or not os.access(target, os.W_OK | os.X_OK):
        raise PermissionError("directory is not writable: " + str(target))
    probe = target / (".pushbutton-write-" + uuid.uuid4().hex)
    try:
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
    finally:
        if probe.exists():
            probe.unlink()


def used_bytes(path):
    total, seen = 0, set()
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                p = pathlib.Path(root) / name
                if p.is_symlink():
                    continue
                info = p.stat()
                identity = (info.st_dev, info.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    total += info.st_size
            except OSError:
                pass
    return total


def repo_roots(cache, repo):
    name = "models--" + repo.replace("/", "--")
    return [pathlib.Path(cache) / prefix / name
            for prefix in ("", "hub", "hf/hub", "llama")]


def metadata(cache, repo):
    key = (canonical(cache), repo)
    if key in METADATA_MEMO:
        return METADATA_MEMO[key]
    local = None
    for root in repo_roots(cache, repo):
        try:
            with open(root / "metadata.json", encoding="utf-8") as stream:
                candidate = json.load(stream)
            if (isinstance(candidate, dict) and isinstance(candidate.get("siblings"), list)
                    and candidate.get("id", repo) == repo):
                local = candidate
                break
        except (OSError, ValueError):
            pass
    offline = any(os.environ.get(name, "").lower() in ("1", "true", "yes", "on")
                  for name in ("HF_HUB_OFFLINE", "PUSHBUTTON_HF_OFFLINE"))
    if offline:
        if local is None:
            raise ValueError(f"Cannot determine download size for {repo} in offline mode; "
                             "provide complete offline metadata.json")
        METADATA_MEMO[key] = local
        return local
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    url = endpoint + "/api/models/" + urllib.parse.quote(repo, safe="/") + "?blobs=true"
    headers = {"User-Agent": "pushbutton-folders/1"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=2 if local is not None else 15) as response:
            data = json.load(response)
        if (not isinstance(data, dict) or not isinstance(data.get("siblings"), list)
                or data.get("id", repo) != repo):
            raise ValueError("invalid HF metadata")
    except (OSError, ValueError) as error:
        if local is None:
            raise ValueError(f"Cannot determine download size for {repo}: {error}; "
                             "provide offline metadata.json or restore HF connectivity") from error
        data = local
    else:
        root = repo_roots(cache, repo)[0]
        staging = root / (".metadata-" + uuid.uuid4().hex + ".json")
        try:
            root.mkdir(parents=True, exist_ok=True)
            fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(data, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, root / "metadata.json")
        except OSError as error:
            print(f"Warning: could not cache HF metadata for {repo}: {error}", file=sys.stderr)
        finally:
            if staging.exists():
                staging.unlink()
    METADATA_MEMO[key] = data
    return data


def download_remaining(spec, cache):
    spec = re.sub(r"^(?:https?://)?(?:huggingface\.co|hf\.co)/", "", spec)
    location, colon, quant = spec.partition(":")
    parts = location.split("/")
    if len(parts) < 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in (".", "..") for p in parts):
        raise ValueError("expected owner/repo:quant or owner/repo/path.gguf[:quant]")
    repo = "/".join(parts[:2])
    explicit = "/".join(parts[2:])
    if explicit and not explicit.lower().endswith(".gguf"):
        raise ValueError("explicit model path must end with .gguf")
    if not explicit and (not colon or not quant):
        raise ValueError("a quant is required for owner/repo:quant")
    if colon and not re.fullmatch(r"[A-Za-z0-9_.-]+", quant):
        raise ValueError("invalid quant")
    data = metadata(cache, repo)
    files = {}
    for entry in data["siblings"]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("rfilename")
        if not isinstance(name, str) or not name.lower().endswith(".gguf"):
            continue
        if name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/")):
            raise ValueError("unsafe filename in HF metadata")
        if name in files:
            raise ValueError("duplicate filename in HF metadata")
        files[name] = entry
    split_re = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)

    def group(name):
        match = split_re.match(name)
        return match.group(1) if match else name

    def matches_quant(name):
        stem = split_re.sub(r"\1", name)
        stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
        # End anchoring avoids Q4 matching Q4_K_M, or Q4_K_M matching XL.
        # UD quants are distinct from their non-UD counterpart.
        match = re.search(r"(?:^|[./_-])(" + re.escape(quant) + r")$", stem, re.I)
        if not match:
            return False
        prefix = stem[:match.start(1)]
        return quant.upper().startswith("UD-") or not prefix.upper().endswith("UD-")

    if explicit:
        if explicit not in files:
            raise ValueError("GGUF file not found: " + explicit)
        selected = [n for n in files if group(n) == group(explicit)]
        if quant and not matches_quant(explicit):
            raise ValueError("explicit GGUF filename does not match requested quant")
    else:
        selected = [n for n in files if matches_quant(n)]
        if not selected:
            raise ValueError("no exact GGUF match for quant " + quant)
        if len({group(n) for n in selected}) != 1:
            raise ValueError("ambiguous GGUF quant; use owner/repo/path.gguf")
    splits = [split_re.match(n) for n in selected]
    if any(splits):
        counts = {int(m.group(3)) for m in splits if m}
        if len(counts) != 1 or not all(splits):
            raise ValueError("inconsistent GGUF splits")
        count = counts.pop()
        if count != len(selected) or {int(m.group(2)) for m in splits} != set(range(1, count + 1)):
            raise ValueError("incomplete GGUF split metadata")
    total = remaining = 0
    for name in selected:
        entry = files[name]
        lfs = entry.get("lfs") or {}
        if not isinstance(lfs, dict):
            raise ValueError("unknown GGUF LFS metadata: " + name)
        size = lfs.get("size", entry.get("size"))
        if type(size) is not int or size <= 0:
            raise ValueError("unknown GGUF size: " + name)
        total += size
        candidates = []
        digest = lfs.get("sha256") or entry.get("blobId")
        for root in repo_roots(cache, repo):
            if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{40,64}", digest):
                candidates.append(root / "blobs" / digest)
            sha = data.get("sha")
            if isinstance(sha, str) and re.fullmatch(r"[a-fA-F0-9]{40,64}", sha):
                candidates.append(root / "snapshots" / sha / name)
                candidates.append(root / sha / name)
        # llama.cpp's flat cache uses owner_repo_filename (slashes become _).
        candidates.append(pathlib.Path(cache) / (repo.replace("/", "_") + "_" + name.replace("/", "_")))
        complete = False
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if ".downloadInProgress" in str(resolved) or resolved.name.endswith(".incomplete"):
                    continue
                if pathlib.Path(str(candidate) + ".aria2").exists():
                    continue
                if candidate.is_file() and candidate.stat().st_size == size:
                    complete = True
                    break
            except OSError:
                pass
        if not complete:
            remaining += size
    print(f"GGUF download: {total / GIB:.2f} GiB total, "
          f"{remaining / GIB:.2f} GiB remaining (complete cache files only).", file=sys.stderr)
    return remaining


def plan_download_required(plan_file, cache):
    with open(plan_file, encoding="utf-8") as stream:
        plan = json.load(stream)
    if not isinstance(plan, dict):
        raise ValueError("download plan must be a JSON object")
    specs = []
    for key in ("servers", "workers"):
        if key not in plan:
            continue
        rows = plan[key]
        if not isinstance(rows, list):
            raise ValueError(f"download plan {key} must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"download plan {key} entries must be objects")
            profile = row.get("profile", row)
            if not isinstance(profile, dict):
                raise ValueError("download plan profile must be an object")
            spec = profile.get("hf_spec")
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(f"download plan {key} entry is missing hf_spec")
            if spec not in specs:
                specs.append(spec)
    if not specs:
        raise ValueError("download plan has no server/worker hf_spec entries")
    remaining = sum(download_remaining(spec, cache) for spec in specs)
    print(f"Download plan: {len(specs)} distinct model spec(s), "
          f"{remaining / GIB:.2f} GiB remaining + 10 GiB safety reserve.",
          file=sys.stderr)
    return remaining + 10 * GIB


def main():
    action, *args = sys.argv[1:]
    if action == "load":
        overrides = args[:3]
        defaults = [canonical(p) for p in DEFAULTS]
        locator = args[3] if len(args) > 3 else ""
        config_home = canonical(overrides[2] or locator) if overrides[2] or locator else defaults[2]
        path = pathlib.Path(config_home) / "folders.json"
        paths = defaults
        status = "valid"
        source = config_home
        try:
            _, paths, source = follow_config(config_home)
        except FileNotFoundError:
            status = "missing"
        except (OSError, ValueError, KeyError, TypeError) as error:
            status = "corrupt"
            print(f"Warning: corrupt folder config {path}: {error}; using defaults.", file=sys.stderr)
        selected = [canonical(override) if override else value
                    for override, value in zip(overrides, paths)]
        distinct(selected)
        changed = status == "valid" and any(
            override and selected[i] != paths[i] for i, override in enumerate(overrides))
        print("\n".join(selected + [status, "1" if changed else "0", source]))
    elif action == "canonical":
        paths = [canonical(p) for p in args]
        distinct(paths)
        print("\n".join(paths))
    elif action in ("check", "save"):
        paths = [canonical(p) for p in args[:3]]
        distinct(paths)
        for path in paths:
            check_directory(path)
        if action == "save":
            dest = pathlib.Path(paths[2]) / "folders.json"
            locator = canonical(args[3]) if len(args) > 3 and args[3] else None
            if locator and locator != paths[2]:
                check_directory(locator)
            created = datetime.datetime.now(datetime.timezone.utc).isoformat()
            try:
                old, _ = read_config(dest)
                created = old["created_at"]
            except (OSError, ValueError, KeyError, TypeError):
                if locator and locator != paths[2]:
                    try:
                        old, _ = read_config(pathlib.Path(locator) / "folders.json")
                        created = old["created_at"]
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
            data = {"schema_version": 1, "folders": dict(zip(KEYS, paths)), "created_at": created}
            persist_config(dest, data)
            if locator and locator != paths[2]:
                persist_config(pathlib.Path(locator) / "folders.json",
                               {**data, "config_locator": True})
    elif action == "free":
        print(free_bytes(args[0]))
    elif action == "space":
        cache, required = args
        required = int(required)
        if required < 0:
            raise ValueError("required cache space must be a nonnegative integer byte count")
        available = free_bytes(cache)
        if available < required:
            print(f"Insufficient cache space at {cache}: {available / GIB:.2f} GiB free, "
                  f"{required / GIB:.2f} GiB required.\n{mount_description(cache)}\n"
                  "Use --local-cache on a larger filesystem, select a smaller quant, "
                  "or free files in the cache.", file=sys.stderr)
            return 1
    elif action == "startup-warning":
        cache = args[0]
        available = free_bytes(cache) / GIB
        if available < 50:
            print(f"Warning: cache at {cache} has only {available:.2f} GiB free "
                  "(less than 50 GiB). Larger GGUF downloads may not fit.\n"
                  f"{mount_description(cache)}\n"
                  "Use --local-cache on a larger filesystem, select a smaller quant, "
                  "or free files in the cache.", file=sys.stderr)
    elif action == "summary":
        state, cache, config, root, verbose, *overrides = args
        print("[pushbutton] System folders")
        print("  Code: " + canonical(root) + " (ROOT)")
        for label, path in (("State", state), ("Cache", cache), ("Config", config)):
            print(f"  {label}: {path} ({free_bytes(path) / GIB:.2f} GiB free)")
        print(f"  Logs: {state}/logs")
        default_locator = pathlib.Path(canonical(DEFAULTS[2])) / "folders.json"
        try:
            locator_data, _ = read_config(default_locator)
            if locator_data.get("config_locator") is True:
                _, locator_paths, _ = follow_config(str(default_locator.parent))
                if locator_paths[2] == canonical(config):
                    print(f"  Config locator: {default_locator} (follows the actual Config file)")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if verbose in ("1", "--verbose", "verbose"):
            for label, path in (("State", state), ("Cache", cache), ("Config", config)):
                print(f"  {label} used: {used_bytes(path) / GIB:.2f} GiB")
            print(f"  Logs used: {used_bytes(state + '/logs') / GIB:.2f} GiB")
            print("  " + mount_description(cache).replace("\n", "\n  "))
        print("  Environment overrides: " + (
            ", ".join(f"{env}={value}" for env, value in
                      zip((*ENVS, "PUSHBUTTON_DIR"), overrides) if value) or "none"))
        print(f"[pushbutton] To customize locations, edit {config}/folders.json or set env vars.")
        print("  Customize folders with CLAUDE_LOCAL_STATE, CLAUDE_LOCAL_CACHE, "
              "PUSHBUTTON_CONFIG_DIR, or --local-cache; PUSHBUTTON_DIR sets the install location.")
    elif action == "download":
        print(download_remaining(args[0], args[1]) + 10 * GIB)
    elif action == "plan-download":
        print(plan_download_required(args[0], canonical(args[1])))
    elif action == "estimate":
        lib, model, quant = args
        result = subprocess.run([sys.executable, str(pathlib.Path(lib) / "claude_local_plan.py"),
                                 "catalogue"], text=True, capture_output=True, check=True)
        catalogue = json.loads(result.stdout)
        profiles = catalogue.get(model, [])
        if not profiles:
            profiles = [p for rows in catalogue.values() for p in rows
                        if p["hf_spec"].lower() == model.lower()]
        if quant:
            profiles = [p for p in profiles if p["quant"].lower() == quant.lower()]
        if not profiles:
            raise ValueError("unknown model/quant planning envelope: " + model)
        print((max(p["required_mib"] for p in profiles) * 1024 ** 2 * 11 + 9) // 10)
    else:
        raise ValueError("unknown folder helper operation: " + action)
    return 0


try:
    sys.exit(main())
except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
    print("Pushbutton folders: " + str(error), file=sys.stderr)
    sys.exit(2 if sys.argv[1] in ("download", "plan-download", "estimate") else 1)
PY
}

# Track what we exported, rather than treating those defaults as user overrides.
# A subsequent deliberate environment change still wins over saved configuration.
_pushbutton_capture_overrides() {
    if [[ ${_PUSHBUTTON_FOLDERS_LOADED:-0} != 1 ]]; then
        _PUSHBUTTON_STATE_OVERRIDE="${CLAUDE_LOCAL_STATE:-}"
        _PUSHBUTTON_CACHE_OVERRIDE="${CLAUDE_LOCAL_CACHE:-}"
        _PUSHBUTTON_CONFIG_OVERRIDE="${PUSHBUTTON_CONFIG_DIR:-}"
        _PUSHBUTTON_CONFIG_LOOKUP="${PUSHBUTTON_CONFIG_DIR:-}"
    else
        [[ ${CLAUDE_LOCAL_STATE:-} == "${_PUSHBUTTON_LAST_STATE:-}" ]] ||
            _PUSHBUTTON_STATE_OVERRIDE="${CLAUDE_LOCAL_STATE:-}"
        [[ ${CLAUDE_LOCAL_CACHE:-} == "${_PUSHBUTTON_LAST_CACHE:-}" ]] ||
            _PUSHBUTTON_CACHE_OVERRIDE="${CLAUDE_LOCAL_CACHE:-}"
        if [[ ${PUSHBUTTON_CONFIG_DIR:-} != "${_PUSHBUTTON_LAST_CONFIG:-}" ]]; then
            _PUSHBUTTON_CONFIG_OVERRIDE="${PUSHBUTTON_CONFIG_DIR:-}"
            _PUSHBUTTON_CONFIG_LOOKUP="${PUSHBUTTON_CONFIG_DIR:-}"
        fi
    fi
    return 0
}

_pushbutton_export_folders() {
    export STATE_DIR CACHE_DIR CONFIG_DIR
    export CLAUDE_LOCAL_STATE="$STATE_DIR" CLAUDE_LOCAL_CACHE="$CACHE_DIR" PUSHBUTTON_CONFIG_DIR="$CONFIG_DIR"
    _PUSHBUTTON_LAST_STATE="$STATE_DIR"
    _PUSHBUTTON_LAST_CACHE="$CACHE_DIR"
    _PUSHBUTTON_LAST_CONFIG="$CONFIG_DIR"
    _PUSHBUTTON_CONFIG_LOOKUP="$CONFIG_DIR"
    _PUSHBUTTON_FOLDERS_LOADED=1
}

load_folder_config() {
    local result
    _pushbutton_capture_overrides
    result="$(_pushbutton_folder_python load "$_PUSHBUTTON_STATE_OVERRIDE" \
        "$_PUSHBUTTON_CACHE_OVERRIDE" "$_PUSHBUTTON_CONFIG_OVERRIDE" \
        "${_PUSHBUTTON_CONFIG_LOOKUP:-}")" || return $?
    {
        IFS= read -r STATE_DIR
        IFS= read -r CACHE_DIR
        IFS= read -r CONFIG_DIR
        IFS= read -r _PUSHBUTTON_FOLDER_CONFIG_STATUS
        IFS= read -r _PUSHBUTTON_FOLDER_OVERRIDES_CHANGED
        IFS= read -r _PUSHBUTTON_FOLDER_CONFIG_SOURCE
    } <<< "$result"
    _pushbutton_export_folders
}

check_folders_exist() {
    _pushbutton_folder_python check "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR"
}

save_folder_config() {
    local result locator=""
    # Also canonicalize choices made by callers or the interactive prompts.
    result="$(_pushbutton_folder_python canonical "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR")" || return $?
    {
        IFS= read -r STATE_DIR
        IFS= read -r CACHE_DIR
        IFS= read -r CONFIG_DIR
    } <<< "$result"
    if [[ -z ${_PUSHBUTTON_CONFIG_OVERRIDE:-} ]]; then
        locator="${_PUSHBUTTON_FOLDER_CONFIG_SOURCE:-${_PUSHBUTTON_CONFIG_LOOKUP:-}}"
    fi
    _pushbutton_folder_python save "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR" "$locator" || return $?
    _pushbutton_export_folders
}

get_free_space() {
    _pushbutton_folder_python free "${1:-$CACHE_DIR}"
}

validate_cache_space() {
    if [[ $# != 1 || -z ${1:-} || -z ${CACHE_DIR:-} ]]; then
        printf 'Pushbutton folders: validate_cache_space requires initialized CACHE_DIR and required bytes\n' >&2
        return 2
    fi
    _pushbutton_folder_python space "$CACHE_DIR" "$1" || return 2
}

validate_download_space() {
    local required location spec="${1:-}" cache="${CACHE_DIR:-}"
    if [[ $# -ge 2 ]]; then
        cache="$1"
        spec="$2"
        if [[ -n ${3:-} ]]; then
            location="${spec#https://}"
            location="${location#http://}"
            if [[ "$location" != *:* ]]; then
                spec="$spec:$3"
            elif [[ "${spec##*:}" != "$3" ]]; then
                printf 'Pushbutton folders: quant %s conflicts with HF spec %s\n' "$3" "$spec" >&2
                return 2
            fi
        fi
    fi
    if [[ -z "$cache" || -z "$spec" || $# -gt 3 ]]; then
        printf 'Pushbutton folders: supply HF_SPEC and initialized CACHE_DIR, or CACHE_DIR HF_SPEC QUANT\n' >&2
        return 2
    fi
    required="$(_pushbutton_folder_python download "$spec" "$cache")" || return 2
    _pushbutton_folder_python space "$cache" "$required" || return 2
}

validate_plan_download_space() {
    local required cache="${2:-${CACHE_DIR:-}}"
    if [[ -z ${1:-} || -z "$cache" || $# -gt 2 ]]; then
        printf 'Pushbutton folders: validate_plan_download_space requires PLAN_FILE and CACHE_DIR\n' >&2
        return 2
    fi
    required="$(_pushbutton_folder_python plan-download "$1" "$cache")" || return 2
    _pushbutton_folder_python space "$cache" "$required" || return 2
}

estimate_model_size() {
    _pushbutton_folder_python estimate "$_PUSHBUTTON_FOLDER_LIB" "${1:?model missing}" "${2:-}"
}

print_folder_summary() {
    _pushbutton_folder_python summary "$STATE_DIR" "$CACHE_DIR" "$CONFIG_DIR" \
        "${PUSHBUTTON_CODE_DIR:-${ROOT:-$_PUSHBUTTON_FOLDER_LIB/..}}" "${1:-0}" \
        "${_PUSHBUTTON_STATE_OVERRIDE:-}" "${_PUSHBUTTON_CACHE_OVERRIDE:-}" \
        "${_PUSHBUTTON_CONFIG_OVERRIDE:-}" "${PUSHBUTTON_DIR:-}"
}

configure_folders() {
    local answer choice interactive=1
    case "${1:-}" in --quiet|quiet|1|true) interactive=0 ;; esac
    load_folder_config || return $?
    if [[ -t 0 && -t 1 && $interactive == 1 ]]; then
        printf '\nPre-install folder summary\n'
        print_folder_summary --verbose || return $?
        printf 'Customize state/cache/config folders? [y/N] '
        IFS= read -r answer || answer=""
        case "$answer" in
            y|Y|yes|YES)
                printf 'State directory [%s]: ' "$STATE_DIR"
                IFS= read -r choice || choice=""
                STATE_DIR="${choice:-$STATE_DIR}"
                printf 'Cache directory [%s]: ' "$CACHE_DIR"
                IFS= read -r choice || choice=""
                CACHE_DIR="${choice:-$CACHE_DIR}"
                printf 'Config directory [%s]: ' "$CONFIG_DIR"
                IFS= read -r choice || choice=""
                CONFIG_DIR="${choice:-$CONFIG_DIR}"
                ;;
        esac
    fi
    save_folder_config
}

initialize_folders() {
    local quiet="${1:-}"
    load_folder_config || return $?
    if [[ ! -f "$CONFIG_DIR/folders.json" ]]; then
        if [[ $_PUSHBUTTON_FOLDER_CONFIG_STATUS == valid ]]; then
            save_folder_config || return $?
        else
            configure_folders || return $?
        fi
    else
        check_folders_exist || return $?
        if [[ $_PUSHBUTTON_FOLDER_CONFIG_STATUS == valid && $_PUSHBUTTON_FOLDER_OVERRIDES_CHANGED == 1 ]]; then
            save_folder_config || return $?
        fi
    fi
    case "$quiet" in
        --quiet|quiet|1|true) ;;
        *) print_folder_summary || return $? ;;
    esac
    _pushbutton_folder_python startup-warning "$CACHE_DIR"
}
