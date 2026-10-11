"""Shared, fail-closed GGUF source resolution using only the standard library."""
import hashlib
import http.client
import ipaddress
import json
import os
import pathlib
import re
import socket
import struct
import urllib.parse
import urllib.request

MAX_PREFIX = 8 * 1024 * 1024
HF_HOST = "huggingface.co"


def validate_url(url, dns=True, signed_redirect=False):
    if not isinstance(url, str) or any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise ValueError("Unsafe URL characters")
    parsed = urllib.parse.urlsplit(url)
    decoded = urllib.parse.unquote(url)
    if any(ord(c) < 32 or ord(c) == 127 for c in decoded) or "\\" in decoded:
        raise ValueError("Unsafe encoded URL characters")
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment or "\\" in url):
        raise ValueError("GGUF URLs must be public HTTPS without credentials or fragments")
    try:
        port = parsed.port or 443
    except ValueError as error:
        raise ValueError("Invalid URL port") from error
    if port != 443:
        raise ValueError("Only HTTPS port 443 is supported")
    for key, _ in urllib.parse.parse_qsl(parsed.query):
        if not signed_redirect and re.search(r"token|auth|key|secret|password|credential|signature", key, re.I):
            raise ValueError("Credential-bearing URL query is not allowed")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("URL host is not public")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        if not dns:
            return url
        try:
            addresses = [ipaddress.ip_address(item[4][0])
                         for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
        except OSError as error:
            raise ValueError("Cannot validate public URL host") from error
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("URL host resolves to a non-public address")
    return url


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_url(newurl, signed_redirect=True)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(newurl).hostname != HF_HOST:
            redirected.remove_header("Authorization")
        return redirected


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        super().connect()
        # Re-check the actual peer before any HTTP bytes are sent (DNS rebinding).
        if not ipaddress.ip_address(self.sock.getpeername()[0]).is_global:
            self.close()
            raise ValueError("HTTPS connected to a non-public address")


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PublicHTTPSConnection, req, context=self._context,
                            check_hostname=self._check_hostname)


def safe_open(request, timeout=30):
    if isinstance(request, str):
        request = urllib.request.Request(request)
    validate_url(request.full_url, signed_redirect=getattr(request, "_validated_redirect", False))
    if urllib.parse.urlsplit(request.full_url).hostname != HF_HOST:
        request.remove_header("Authorization")
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _SafeRedirect(),
                                       _PublicHTTPSHandler()).open(request, timeout=timeout)


def hf_headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        home = os.environ.get("HF_HOME", str(
            pathlib.Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")) / "huggingface"))
        path = pathlib.Path(os.path.expandvars(os.environ.get(
            "HF_TOKEN_PATH", str(pathlib.Path(home) / "token")))).expanduser()
        if path.is_file():
            token = path.read_text().strip()
    if token and any(ord(c) < 32 or ord(c) == 127 for c in token):
        raise ValueError("Invalid HF token")
    return {"Authorization": "Bearer " + token} if token else {}


def _filename(name):
    path = pathlib.PurePosixPath(name)
    if (not name or path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))
            or "\\" in name or any(ord(c) < 32 or ord(c) == 127 for c in name)):
        raise ValueError("Unsafe GGUF filename")
    return name


def parse_source(text):
    if not isinstance(text, str) or not text or text != text.strip() or any(
            ord(c) < 32 or ord(c) == 127 for c in text):
        raise ValueError("Invalid GGUF source")
    if text.startswith("https://"):
        validate_url(text, dns=False)
        parsed = urllib.parse.urlsplit(text)
        path = urllib.parse.unquote(parsed.path)
        if not path.lower().endswith(".gguf"):
            raise ValueError("Source URL must end with .gguf")
        if parsed.hostname == HF_HOST:
            match = re.fullmatch(r"/([\w.-]+/[\w.-]+)/resolve/([^/]+)/(.+)", path)
            if not match:
                raise ValueError("Expected Hugging Face /resolve/REV/path.gguf URL")
            repo, revision, selector = match.groups()
            if any(p in (".", "..") for p in repo.split("/")):
                raise ValueError("Unsafe Hugging Face repository")
            _filename(selector)
            if revision in (".", "..") or not re.fullmatch(r"[\w.-]+", revision):
                raise ValueError("Unsafe Hugging Face revision")
            return {"repo": repo, "revision": revision, "selector": selector, "url": text}
        _filename(path.rsplit("/", 1)[-1])
        return {"repo": None, "revision": "main", "selector": None, "url": text}
    value = text.removeprefix("hf:")
    location, colon, selector = value.partition(":")
    parts = location.split("/")
    if len(parts) < 2 or any(not re.fullmatch(r"[\w.-]+", p) or p in (".", "..")
                             for p in parts[:2]):
        raise ValueError("Expected owner/repo:quant or owner/repo/path.gguf")
    if len(parts) > 2:
        if colon:
            raise ValueError("Exact GGUF paths do not take a quant suffix")
        selector = _filename("/".join(parts[2:]))
        if not selector.lower().endswith(".gguf"):
            raise ValueError("Source path must end with .gguf")
    elif not colon:
        selector = "Q4_K_M"
    elif not selector:
        raise ValueError("Hugging Face source needs a quant selector")
    if not selector.lower().endswith(".gguf") and not re.fullmatch(
            r"[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*", selector):
        raise ValueError("Invalid quant selector")
    _filename(selector)
    return {"repo": "/".join(parts[:2]), "revision": "main", "selector": selector, "url": None}


def is_source(text):
    try:
        parse_source(text)
        return True
    except (ValueError, TypeError):
        return False


def fetch_metadata(repo, revision="main"):
    url = (f"https://{HF_HOST}/api/models/{repo}/revision/"
           f"{urllib.parse.quote(revision, safe='')}?blobs=true")
    with safe_open(urllib.request.Request(url, headers=hf_headers())) as response:
        data = response.read(MAX_PREFIX + 1)
    if len(data) > MAX_PREFIX:
        raise ValueError("Hugging Face metadata exceeds safety limit")
    result = json.loads(data)
    if not isinstance(result, dict) or not isinstance(result.get("siblings"), list):
        raise ValueError("Invalid Hugging Face metadata")
    return result


def cache_directory(cache, resolved):
    root = pathlib.Path(cache).expanduser().resolve()
    if resolved["repo"]:
        return root / ("models--" + resolved["repo"].replace("/", "--")) / resolved["revision"]
    return root / "urls" / resolved["revision"]


def resolve_files(text, metadata=None):
    source = parse_source(text)
    if source["repo"]:
        # Import lazily: downloader shares this resolver but owns selector validation.
        from pushbutton_download import select_files
        data = metadata if metadata is not None else fetch_metadata(source["repo"], source["revision"])
        revision = data.get("sha", "")
        if not re.fullmatch(r"[a-fA-F0-9]{40,64}", revision):
            raise ValueError("Cannot confirm Hugging Face model revision")
        if re.fullmatch(r"[a-fA-F0-9]{40,64}", source["revision"]) and revision != source["revision"]:
            raise ValueError("Hugging Face metadata does not match pinned revision")
        files = select_files(data, source["selector"])
        for item in files:
            item["url"] = (f"https://{HF_HOST}/{source['repo']}/resolve/{revision}/"
                           + urllib.parse.quote(item["name"]))
    else:
        url = source["url"]
        validate_url(url)
        name = _filename(urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]))
        if re.fullmatch(r".*-\d{5}-of-\d{5}\.gguf", name, re.I):
            raise ValueError("Public URL shards require a Hugging Face repository inventory")
        with safe_open(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as response:
            size = _response_size(response)
            etag = response.headers.get("ETag", "")
            modified = response.headers.get("Last-Modified", "")
        revision = hashlib.sha256(
            (url + "\n" + str(size) + "\n" + etag + "\n" + modified).encode()).hexdigest()
        files = [{"name": name, "size": size, "url": url, "etag": etag}]
    return {**source, "revision": revision, "files": files,
            "weight_bytes": sum(f["size"] for f in files),
            "download_spec": files[0]["url"]}


def _response_size(response):
    content_range = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes \d+-\d+/(\d+)", content_range)
    size = int(match[1]) if match else int(response.headers.get("Content-Length", "0"))
    if not match and response.status != 200:
        raise ValueError("Cannot confirm remote GGUF size")
    if size <= 0:
        raise ValueError("Cannot confirm remote GGUF size")
    return size


class _NeedMore(ValueError):
    pass


def read_gguf_metadata(data):
    """Read GGUF v2/v3 metadata only; never parse tensors or execute model code."""
    offset = 0

    def take(size):
        nonlocal offset
        if size > MAX_PREFIX or offset + size > MAX_PREFIX:
            raise ValueError("GGUF metadata exceeds safety limit")
        if offset + size > len(data):
            raise _NeedMore("Incomplete GGUF metadata prefix")
        result = data[offset:offset + size]
        offset += size
        return result

    def number(fmt):
        return struct.unpack("<" + fmt, take(struct.calcsize("<" + fmt)))[0]

    def string():
        return take(number("Q")).decode("utf-8")

    formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
               6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(kind, keep=True):
        if kind in formats:
            return number(formats[kind])
        if kind == 8:
            return string()
        if kind == 9:
            element, count = number("I"), number("Q")
            if count > MAX_PREFIX or element == 9:
                raise ValueError("Unsupported GGUF metadata array")
            if element in formats and not keep:
                take(count * struct.calcsize("<" + formats[element]))
                return None
            values = [] if keep else None
            for _ in range(count):
                item = value(element, keep)
                if keep:
                    values.append(item)
            return values
        raise ValueError("Unsupported GGUF metadata value type")

    if take(4) != b"GGUF" or number("I") not in (2, 3):
        raise ValueError("Expected GGUF v2/v3 weights")
    number("Q")
    count = number("Q")
    if count > 100000:
        raise ValueError("GGUF metadata count exceeds safety limit")
    result = {}
    for _ in range(count):
        key, kind = string(), number("I")
        keep = not key.startswith("tokenizer.")
        item = value(kind, keep)
        if keep:
            if key in result:
                raise ValueError("Duplicate GGUF metadata key")
            result[key] = item
    return result


def architecture(metadata):
    arch = metadata.get("general.architecture")
    dense = {"llama", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "gemma", "gemma2",
             "gemma3", "phi2", "phi3", "falcon", "stablelm", "starcoder2", "internlm2",
             "olmo", "olmo2", "mistral"}
    hybrid = {"qwen3next", "qwen35", "qwen35moe"}
    recurrent = {"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7"}
    if arch not in dense | hybrid | recurrent:
        raise ValueError(f"Cannot conservatively derive KV/context for GGUF architecture {arch!r}")

    def integer(key, default=None):
        value = metadata.get(arch + "." + key, default)
        if type(value) is not int or value <= 0:
            raise ValueError("Cannot derive GGUF KV/context: missing or invalid " + key)
        return value

    context = integer("context_length")
    layers = integer("block_count")
    if arch in recurrent:
        return {"context": context, "attn_layers": 0, "kv_dim": 0,
                "kv_key_dim": 0, "kv_value_dim": 0,
                "architecture_source": "gguf:" + arch}
    heads = integer("attention.head_count")
    kv_heads = integer("attention.head_count_kv")
    embedding = integer("embedding_length")
    if embedding % heads:
        raise ValueError("Cannot derive GGUF attention head dimensions")
    explicit_dimensions = arch in hybrid or arch.startswith("gemma")
    key_dim = integer("attention.key_length", None if explicit_dimensions else embedding // heads)
    value_dim = integer("attention.value_length", None if explicit_dimensions else key_dim)
    if arch in hybrid:
        interval = integer("full_attention_interval", metadata.get(
            arch + ".attention.full_attention_interval"))
        layers = (layers + interval - 1) // interval
        if not layers:
            raise ValueError("Cannot derive hybrid full-attention layers")
    # kv_dim includes KV heads and averages K/V, matching planners' 2 * kv_dim.
    return {"context": context, "attn_layers": layers,
            "kv_dim": kv_heads * (key_dim + value_dim) / 2,
            "kv_key_dim": kv_heads * key_dim,
            "kv_value_dim": kv_heads * value_dim,
            "architecture_source": "gguf:" + arch}


def resolve_source(text):
    parsed = parse_source(text)
    if parsed["selector"] and "," in parsed["selector"]:
        raise ValueError("Architecture planning requires a single GGUF model variant")
    resolved = resolve_files(text)
    item = resolved["files"][0]
    cache = os.environ.get("PUSHBUTTON_LOCAL_CACHE") or os.environ.get("HF_HUB_CACHE")
    local = cache_directory(cache, resolved) / item["name"] if cache else None
    limit = min(128 * 1024, item["size"])
    while True:
        if local and local.is_file() and local.stat().st_size == item["size"] and not pathlib.Path(
                str(local) + ".aria2").exists():
            with local.open("rb") as stream:
                prefix = stream.read(limit)
        else:
            headers = hf_headers() if resolved["repo"] else {}
            headers["Range"] = f"bytes=0-{limit - 1}"
            if item.get("etag"):
                headers["If-Match"] = item["etag"]
            with safe_open(urllib.request.Request(item["url"], headers=headers)) as response:
                if _response_size(response) != item["size"]:
                    raise ValueError("GGUF size changed during metadata resolution")
                if response.status == 206 and not response.headers.get("Content-Range", "").startswith("bytes 0-"):
                    raise ValueError("Invalid GGUF prefix range")
                prefix = response.read(limit)
        try:
            info = architecture(read_gguf_metadata(prefix))
            break
        except _NeedMore:
            if limit >= min(MAX_PREFIX, item["size"]):
                raise ValueError("Cannot derive KV/context within bounded GGUF metadata prefix")
            limit = min(limit * 4, MAX_PREFIX, item["size"])
    selector = resolved["selector"] or item["name"]
    quant = re.search(r"(?:^|[-./])((?:UD-)?(?:IQ|Q|BF|F)\d[A-Za-z0-9_]*)",
                      selector, re.I)
    return {**resolved, **info, "quant": quant[1].upper() if quant else "custom"}


def download(text, cache_dir):
    """Download an exact source safely and return its first GGUF shard path."""
    from pushbutton_download import download_safe, reuse_blobs, validate_space
    resolved = resolve_files(text)
    directory = cache_directory(cache_dir, resolved)
    directory.mkdir(parents=True, exist_ok=True)
    reuse_blobs(directory, resolved["files"])
    manifest = {**resolved, "directory": str(directory)}
    validate_space(manifest)
    download_safe(manifest)
    return directory / resolved["files"][0]["name"]
