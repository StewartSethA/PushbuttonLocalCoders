#!/usr/bin/env python3
"""Resolve exact GGUF files and enforce storage policy for both download paths."""
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gguf_source


def display_name(name, limit=72):
    return name if len(name) <= limit else name[:limit - 3] + "..."


def complete(path, size):
    return (path.is_file() and path.stat().st_size == size
            and not pathlib.Path(str(path) + ".aria2").exists())


def select_files(metadata, quant):
    if not isinstance(quant, str) or not quant or any(ord(c) < 32 or ord(c) == 127 for c in quant):
        raise ValueError("Invalid GGUF selector")
    if not quant.lower().endswith(".gguf") and not re.fullmatch(
            r"[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*", quant):
        raise ValueError("Invalid GGUF selector")
    if "," in quant:
        files = [f for variant in quant.split(",") for f in select_files(metadata, variant)]
        return list({f["name"]: f for f in files}.values())
    matches = []
    exact_shard = re.fullmatch(r"(.*)-\d{5}-of-\d{5}\.gguf", quant)
    for item in metadata.get("siblings", []):
        name = item.get("rfilename", item.get("name", ""))
        path = pathlib.PurePosixPath(name)
        if (not name or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))
                or any(ord(c) < 32 or ord(c) == 127 for c in name) or "\\" in name):
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
        elif quant.lower().endswith(".gguf"):
            if name != quant and ("/" in quant or base != quant):
                continue
        elif not re.search(pattern, base, re.IGNORECASE):
            continue
        if not quant.upper().startswith("UD-") and re.search(
                r"UD-" + re.escape(quant) + r"(?:[.-]|$)", base, re.IGNORECASE):
            continue
        size = item.get("size") or (item.get("lfs") or {}).get("size")
        if type(size) is not int or size <= 0:
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
        partial = pathlib.Path(str(path) + ".part")
        stable_resume = bool(manifest.get("repo") or item.get("etag"))
        if (stable_resume and partial.is_file() and partial.stat().st_size < item["size"]
                and os.path.commonpath((str(directory.resolve()), str(partial.resolve()))) == str(directory.resolve())):
            stat = partial.stat()
            allocated = min(item["size"], getattr(stat, "st_blocks", 0) * 512)
        elif reuse_partial and path.is_file() and pathlib.Path(str(path) + ".aria2").exists():
            stat = path.stat()
            allocated = min(item["size"], getattr(stat, "st_blocks", 0) * 512)
        remaining += item["size"] - allocated
    if not remaining:
        return
    disk = shutil.disk_usage(directory)
    reserve = 10 * 1024**3
    if remaining + reserve > disk.free:
        raise ValueError(
            f"Insufficient disk space in {directory}: need {remaining / 1024**3:.1f} GiB "
            f"+ {reserve / 1024**3:.1f} GiB reserve; free {disk.free / 1024**3:.1f} GiB. "
            "Use --local-cache /bigger/disk, choose a smaller quant, or free space.")


def reuse_blobs(directory, files):
    root = os.path.realpath(directory)
    for item in files:
        target = directory / item["name"]
        resolved_target = os.path.realpath(target)
        try:
            contained = os.path.commonpath((root, resolved_target)) == root
        except ValueError:
            contained = False
        if not contained:
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
    if any(ord(c) < 32 for c in str(pathlib.Path(cache).expanduser().resolve())):
        raise ValueError("Model folder must not contain control characters")
    resolved = gguf_source.resolve_files(spec)
    repo, revision, files = resolved["repo"], resolved["revision"], resolved["files"]
    headers = gguf_source.hf_headers() if repo else {}
    directory = gguf_source.cache_directory(cache, resolved)
    directory.mkdir(parents=True, exist_ok=True)
    reuse_blobs(directory, files)
    classic_spec = bool(re.fullmatch(r"[\w.-]+/[\w.-]+:[A-Za-z0-9_.-]+", spec))
    manifest = {"repo": repo, "spec": spec, "revision": revision,
                "directory": str(directory), "cache": str(pathlib.Path(cache).expanduser().resolve()),
                "files": files, "model_path": str(directory / files[0]["name"]),
                "safe_download": not classic_spec or bool(headers)}
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
            urls.write(item["url"] + "\n")
            urls.write(f"  dir={target.parent}\n  out={target.name}\n")
            digest = item.get("sha256")
            if isinstance(digest, str) and re.fullmatch(r"[a-fA-F0-9]{64}", digest):
                urls.write(f"  checksum=sha-256={digest}\n")
            if headers:
                urls.write("  header=Authorization: " + headers["Authorization"] + "\n")
            names.write(name + "\n")
    total = sum(f["size"] for f in files)
    label = "Downloading" if pending else "Using cached"
    print(f"[pushbutton] {label} {display_name(spec)} ({total / 1024**3:.1f} GiB total)\n"
          f"[pushbutton] Model folder: {directory}", file=sys.stderr)
    return manifest


def download_safe(manifest):
    """Sequential resume with validated redirects and credentials scoped to HF."""
    directory = pathlib.Path(manifest["directory"])
    reuse_blobs(directory, manifest["files"])
    for item in manifest["files"]:
        target = directory / item["name"]
        if complete(target, item["size"]):
            continue
        partial = pathlib.Path(str(target) + ".part")
        if os.path.commonpath((str(directory), str(partial.resolve()))) != str(directory):
            raise ValueError("Partial file escapes cache directory")
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset >= item["size"] or (not manifest["repo"] and not item.get("etag")):
            partial.unlink(missing_ok=True)
            offset = 0
        headers = gguf_source.hf_headers() if manifest["repo"] else {}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        if item.get("etag"):
            headers["If-Match"] = item["etag"]
        with gguf_source.safe_open(urllib.request.Request(item["url"], headers=headers)) as response:
            if gguf_source._response_size(response) != item["size"]:
                raise ValueError("GGUF size changed; resolve source again")
            append = offset and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Invalid resumed GGUF range")
            written = offset if append else 0
            with partial.open("ab" if append else "wb") as stream:
                while True:
                    chunk = response.read(min(1024 * 1024, item["size"] - written + 1))
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > item["size"]:
                        raise ValueError("GGUF response exceeds confirmed size")
                    stream.write(chunk)
            if written != item["size"]:
                raise ValueError("GGUF download incomplete; re-run to resume")
        digest = item.get("sha256")
        if digest:
            checksum = hashlib.sha256()
            with partial.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest().lower() != digest.lower():
                partial.unlink()
                raise ValueError("GGUF checksum mismatch")
        os.replace(partial, target)
        pathlib.Path(str(target) + ".aria2").unlink(missing_ok=True)
    verify(manifest)


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
    elif action == "download":
        download_safe(manifest)
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
