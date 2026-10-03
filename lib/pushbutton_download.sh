#!/usr/bin/env bash
# Shared pre-startup model downloads; progress goes to stderr, paths to stdout.
PUSHBUTTON_DOWNLOAD_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pushbutton_download.py"

ensure_aria2c() {
    if [[ -z "${_PUSHBUTTON_ARIA_CHECKED:-}" ]]; then
        _PUSHBUTTON_ARIA_CHECKED=1
        if ! command -v aria2c >/dev/null 2>&1 && [[ "$(uname -s)" == Linux ]] && command -v timeout >/dev/null 2>&1; then
            local -a elevate=()
            if [[ $EUID -ne 0 ]]; then elevate=(sudo -n); fi
            if command -v apt-get >/dev/null 2>&1; then
                timeout --kill-after=5s 30s "${elevate[@]}" env DEBIAN_FRONTEND=noninteractive \
                    apt-get -o DPkg::Lock::Timeout=0 -o Acquire::Retries=0 -o Acquire::http::Timeout=10 \
                    install -y aria2 </dev/null >/dev/null 2>&1 || true
            elif command -v dnf >/dev/null 2>&1; then
                timeout --kill-after=5s 30s "${elevate[@]}" dnf install -y aria2 </dev/null >/dev/null 2>&1 || true
            fi
        fi
        if command -v aria2c >/dev/null 2>&1; then
            printf '[pushbutton] Enabling fast parallel downloads with aria2c\n' >&2
        else
            printf '[pushbutton] Using sequential downloads (aria2c not available)\n' >&2
        fi
    fi
    command -v aria2c >/dev/null 2>&1
}

_pushbutton_prepare_download() {
    PUSHBUTTON_NO_PARALLEL="${PUSHBUTTON_NO_PARALLEL:-0}" \
        python3 "$PUSHBUTTON_DOWNLOAD_PY" prepare "$1" "$2" "$3" || return
    # Allow the folder-management layer to impose additional storage policy.
    if declare -F validate_download_space >/dev/null; then
        validate_download_space "$1" "$2" || return
    fi
}

download_model_with_aria2c() (
    trap 'exit 130' INT
    trap 'exit 143' TERM
    local hf_spec="$1" cache_dir="$2" work="${3:-}" own_work=0
    if [[ -z "$work" ]]; then
        work="$(mktemp -d)" || return
        own_work=1
        trap 'rm -rf -- "$work"' EXIT
        _pushbutton_prepare_download "$hf_spec" "$cache_dir" "$work" || return
    fi
    local dir
    dir="$(python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" directory)" || return
    # Keep credentials in the private input file, not in process arguments.
    aria2c --dir="$dir" --input-file="$work/urls" \
        --max-concurrent-downloads=4 --max-connection-per-server=3 --split=3 \
        --min-split-size=5M --allow-overwrite=true --auto-file-renaming=false \
        --file-allocation=prealloc --continue=true --summary-interval=1 \
        --max-tries=5 --retry-wait=10 --connect-timeout=30 --timeout=60 \
        --console-log-level=warn >&2 || return
    python3 "$PUSHBUTTON_DOWNLOAD_PY" verify "$work/manifest.json" || return
    if [[ $own_work == 1 ]]; then
        python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" model_path
    fi
)

download_model_with_hf_cli() (
    trap 'exit 130' INT
    trap 'exit 143' TERM
    local hf_spec="$1" cache_dir="$2" work="${3:-}" own_work=0
    if [[ -z "$work" ]]; then
        work="$(mktemp -d)" || return
        own_work=1
        trap 'rm -rf -- "$work"' EXIT
        _pushbutton_prepare_download "$hf_spec" "$cache_dir" "$work" || return
    fi
    local cli dir repo revision file venv
    if command -v hf >/dev/null 2>&1; then
        cli="$(command -v hf)"
    else
        venv="${STATE_DIR:-${CLAUDE_LOCAL_STATE:-$HOME/.local/share/pushbutton/claude-local}}/download-venv"
        if [[ ! -x "$venv/bin/hf" ]]; then
            printf '[pushbutton] Installing Hugging Face download CLI in %s\n' "$venv" >&2
            python3 -m venv "$venv" >&2 || return
            "$venv/bin/python" -m pip install --disable-pip-version-check 'huggingface_hub==2.1.1' >&2 || return
        fi
        cli="$venv/bin/hf"
    fi
    dir="$(python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" directory)" || return
    repo="$(python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" repo)" || return
    revision="$(python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" revision)" || return
    printf '[pushbutton] Downloading %s (sequential, slower)\n' "$hf_spec" >&2
    while IFS= read -r file; do
        HF_HUB_DISABLE_PROGRESS_BARS=0 HF_HUB_DISABLE_XET=1 \
            "$cli" download "$repo" "$file" --revision "$revision" --local-dir "$dir" >&2 || return
        rm -f -- "$dir/$file.aria2" || return
    done < "$work/files"
    python3 "$PUSHBUTTON_DOWNLOAD_PY" verify "$work/manifest.json" || return
    if [[ $own_work == 1 ]]; then
        python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" model_path
    fi
)

download_model_fast() {
    local hf_spec="$1" cache_dir="$2" output_var="$3" work result status=0 mode=sequential started
    [[ "$output_var" =~ ^[a-zA-Z_][a-zA-Z_0-9]*$ ]] || return 1
    work="$(mktemp -d)" || return
    if ! _pushbutton_prepare_download "$hf_spec" "$cache_dir" "$work"; then
        rm -rf -- "$work"
        return 1
    fi
    started="$SECONDS"
    if [[ -s "$work/files" ]]; then
        if [[ "${PUSHBUTTON_NO_PARALLEL:-0}" != 1 ]] && ensure_aria2c; then
            mode=aria2c
            download_model_with_aria2c "$hf_spec" "$cache_dir" "$work" || status=$?
            if [[ $status -ne 0 && $status -ne 130 && $status -ne 143 ]]; then
                printf '[pushbutton] Parallel download failed; trying Hugging Face sequential download.\n' >&2
                mode=sequential
                status=0
                # HF CLI cannot resume aria2 chunks; reserve space for its own temporary files.
                python3 "$PUSHBUTTON_DOWNLOAD_PY" space "$work/manifest.json" &&
                    download_model_with_hf_cli "$hf_spec" "$cache_dir" "$work" || status=$?
            fi
        else
            python3 "$PUSHBUTTON_DOWNLOAD_PY" space "$work/manifest.json" &&
                download_model_with_hf_cli "$hf_spec" "$cache_dir" "$work" || status=$?
        fi
        if [[ $status -eq 0 ]]; then
            python3 "$PUSHBUTTON_DOWNLOAD_PY" log "$work/manifest.json" "$((SECONDS-started))" "$mode" || status=$?
        fi
    fi
    if [[ $status -eq 0 ]]; then
        result="$(python3 "$PUSHBUTTON_DOWNLOAD_PY" field "$work/manifest.json" model_path)" || status=$?
        [[ $status -ne 0 ]] || printf -v "$output_var" '%s' "$result"
    fi
    rm -rf -- "$work"
    return "$status"
}
