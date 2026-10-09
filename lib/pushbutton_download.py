#!/usr/bin/env python3
"""Resolve exact GGUF files and enforce storage policy for both download paths."""
import datetime
import json
import os
import pathlib
import re
import shutil
import sys
import urllib.parse
import urllib.request


def display_name(name, limit=72):
    return name if len(name) <= limit else name[:limit - 3] + "..."


def complete(path, size):
    return (path.is_file() and path.stat().st_size == size
            and not pathlib.Path(str(path) + ".aria2").exists())


def select_files(metadata, quant):
    if "," in quant:
        files = [f for variant in quant.split(",") for f in select_files(metadata, variant)]
        return list({f["name"]: f for f in files}.values())
    matches = []
    exact_shard = re.fullmatch(r"(.*)-\d{5}-of-\d{5}\.gguf", quant)
    for item in metadata.get("siblings", []):
        name = item.get("rfilename", item.get("name", ""))
        path = pathlib.PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts
                or any(ord(c) < 32 for c in name) or "\\" in name):
            raise ValueError("Unsafe filename in Hugging Face metadata")
        if not name.lower().endswith(".gguf") or "mmproj" in name.lower():
            continue
        # Token boundaries prevent Q4_K matching Q4_K_M or IQ3 matching UD-IQ3.
        base = path.name
        # Optional MTP builds require an explicit variant or filename selector.
        if (re.search(r"-mtp(?:-\d{5}-of-\d{5})?\.gguf$", base, re.IGNORECASE)
                and not re.search(r"(?:^|-)mtp(?:[.-]|$)", quant, re.IGNORECASE)):
            continue
        pattern = r"(?:^|[-.])" + re.escape(quant) + r"(?:[.-]|$)"
        shard = re.fullmatch(r"(.*)-\d{5}-of-\d{5}\.gguf", name)
        if exact_shard and shard and exact_shard[1] == shard[1]:
            pass
        elif base != quant and name != quant and not re.search(pattern, base, re.IGNORECASE):
            continue
        if not quant.upper().startswith("UD-") and re.search(
                r"UD-" + re.escape(quant) + r"(?:[.-]|$)", base, re.IGNORECASE):
            continue
        size = item.get("size") or (item.get("lfs") or {}).get("size")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Cannot confirm download size for {name}")
        matches.append({"name": name, "size": size, "sha256": (item.get("lfs") or {}).get("sha256")})
    if not matches:
        raise ValueError(f"No GGUF files found for quant {quant}")
    matches.sort(key=lambda f: f["name"])
    # Accept one model, or one complete set of GGUF shards, never ambiguous variants.
    if len(matches) > 1 or re.fullmatch(r".*-\d{5}-of-\d{5}\.gguf", matches[0]["name"]):
        shards = [re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", f["name"])
                  for f in matches]
        if (not all(shards) or len({m[1] for m in shards}) != 1
                or {int(m[3]) for m in shards} != {len(matches)}
                or {int(m[2]) for m in shards} != set(range(1, len(matches) + 1))):
            raise ValueError("Ambiguous or incomplete GGUF shard set; specify an exact filename")
    return matches


def validate_space(manifest, reuse_partial=False):
    directory = pathlib.Path(manifest["directory"])
    remaining = 0
    for item in manifest["files"]:
        path = directory / item["name"]
        if complete(path, item["size"]):
            continue
        allocated = 0
        if reuse_partial and path.is_file() and pathlib.Path(str(path) + ".aria2").exists():
            stat = path.stat()
            allocated = min(item["size"], getattr(stat, "st_blocks", 0) * 512)
        remaining += item["size"] - allocated
    if not remaining:
        return
    disk = shutil.disk_usage(directory)
    reserve = max(10 * 1024**3, int(disk.total * 0.05))
    if remaining + reserve > disk.free:
        raise ValueError(
            f"Insufficient disk space in {directory}: need {remaining / 1024**3:.1f} GiB "
            f"+ {reserve / 1024**3:.1f} GiB reserve; free {disk.free / 1024**3:.1f} GiB. "
            "Use --local-cache /bigger/disk, choose a smaller quant, or free space.")


def reuse_blobs(directory, files):
    for item in files:
        target = directory / item["name"]
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Model filename escapes cache directory")
        if target.exists():
            continue
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            continue
        blob = directory.parent / "blobs" / digest
        if complete(blob, item["size"]):
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(blob, target)
            except OSError:
                # Never copy multi-GB weights just to migrate a cache.
                continue
            print(f"[pushbutton] Reusing cached {display_name(item['name'])}", file=sys.stderr)


def prepare(spec, cache, work):
    repo, sep, quant = spec.partition(":")
    if not sep or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or not quant:
        raise ValueError("Expected Hugging Face spec owner/repo:quant (or exact GGUF filename)")
    if any(ord(c) < 32 for c in str(pathlib.Path(cache).expanduser().resolve())):
        raise ValueError("Model folder must not contain control characters")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        hf_home = os.environ.get("HF_HOME", str(
            pathlib.Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface"))
        token_file = pathlib.Path(os.path.expandvars(os.environ.get(
            "HF_TOKEN_PATH", str(pathlib.Path(hf_home) / "token")))).expanduser()
        if token_file.is_file():
            token = token_file.read_text().strip()
    if token and any(ord(c) < 32 for c in token):
        raise ValueError("Invalid HF token")
    headers = {"Authorization": "Bearer " + token} if token else {}
    url = f"https://huggingface.co/api/models/{repo}/revision/main?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        metadata = json.load(response)
    revision = metadata.get("sha", "")
    if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
        raise ValueError("Cannot confirm Hugging Face model revision")
    files = select_files(metadata, quant)
    directory = pathlib.Path(cache).expanduser().resolve() / ("models--" + repo.replace("/", "--")) / revision
    directory.mkdir(parents=True, exist_ok=True)
    reuse_blobs(directory, files)
    manifest = {"repo": repo, "spec": spec, "revision": revision,
                "directory": str(directory), "cache": str(pathlib.Path(cache).expanduser().resolve()),
                "files": files, "model_path": str(directory / files[0]["name"])}
    validate_space(manifest, reuse_partial=os.environ.get("PUSHBUTTON_NO_PARALLEL") != "1")
    pending = [f for f in files if not complete(directory / f["name"], f["size"])]
    manifest["download_bytes"] = sum(f["size"] for f in pending)
    work = pathlib.Path(work)
    (work / "manifest.json").write_text(json.dumps(manifest))
    with (work / "urls").open("w") as urls, (work / "files").open("w") as names:
        (work / "urls").chmod(0o600)
        for item in pending:
            name = item["name"]
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or pathlib.Path(str(target) + ".aria2").exists():
                print(f"[pushbutton] Resuming {display_name(name)} (partial download found)", file=sys.stderr)
            urls.write(f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}\n")
            urls.write(f"  dir={target.parent}\n  out={target.name}\n")
            digest = item.get("sha256")
            if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                urls.write(f"  checksum=sha-256={digest}\n")
            if token:
                urls.write("  header=Authorization: " + "Bearer " + token + "\n")
            names.write(name + "\n")
    total = sum(f["size"] for f in files)
    label = "Downloading" if pending else "Using cached"
    print(f"[pushbutton] {label} {display_name(spec)} ({total / 1024**3:.1f} GiB total)\n"
          f"[pushbutton] Model folder: {directory}", file=sys.stderr)
    return manifest


def verify(manifest):
    directory = pathlib.Path(manifest["directory"])
    for item in manifest["files"]:
        if not complete(directory / item["name"], item["size"]):
            raise ValueError(f"Download incomplete: {item['name']}; re-run to resume")


def log_download(manifest, elapsed, mode):
    size = manifest["download_bytes"]
    event = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "model": manifest["spec"], "size": sum(f["size"] for f in manifest["files"]),
             "download_bytes": size, "seconds": elapsed,
             "speed_bytes_per_second": size / max(1, elapsed), "method": mode}
    with (pathlib.Path(manifest["cache"]) / ".download-log").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def main():
    action, *args = sys.argv[1:]
    if action == "prepare":
        prepare(*args)
        return
    manifest = json.loads(pathlib.Path(args[0]).read_text())
    if action == "field":
        print(manifest[args[1]])
    elif action == "verify":
        verify(manifest)
    elif action == "space":
        validate_space(manifest)
    elif action == "pending":
        directory = pathlib.Path(manifest["directory"])
        pathlib.Path(args[1]).write_text("".join(
            item["name"] + "\n" for item in manifest["files"]
            if not complete(directory / item["name"], item["size"])))
    elif action == "log":
        log_download(manifest, int(args[1]), args[2])
    else:
        raise ValueError(f"Unknown download action: {action}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"[pushbutton] {exc}", file=sys.stderr)
        sys.exit(1)
