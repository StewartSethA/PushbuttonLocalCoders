#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, shutil, sys
from pathlib import Path

# Keep this marker stable across lab releases so re-applying a newer harness to a
# tree patched by an older one is safe and detectable.
MARK = 'KNL_LLAMA_LAB_V3'
PATCH_REV = '2026-08-22.1'
FILES = [
    Path('ggml/CMakeLists.txt'),
    Path('ggml/src/CMakeLists.txt'),
    Path('ggml/src/ggml-cpu/CMakeLists.txt'),
    Path('ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp'),
    Path('ggml/src/ggml-cpu/arch/x86/quants.c'),
]


def die(s: str):
    print('ERROR:', s, file=sys.stderr)
    raise SystemExit(2)


def one_match(t: str, pattern: str, label: str, flags: int = 0) -> re.Match[str]:
    ms = list(re.finditer(pattern, t, flags))
    if len(ms) != 1:
        die(f'{label}: expected one semantic anchor, found {len(ms)}; upstream changed')
    return ms[0]


def insert_after_match(t: str, m: re.Match[str], payload: str) -> str:
    pos = m.end()
    # Semantic anchors are normally complete lines. Preserve exactly one newline
    # between the original line and the inserted block.
    if pos < len(t) and t[pos:pos + 1] == '\n':
        pos += 1
    return t[:pos] + payload + t[pos:]


def insert_before_match(t: str, m: re.Match[str], payload: str) -> str:
    return t[:m.start()] + payload + t[m.start():]


def find_function_chunk(t: str, signature: str) -> tuple[int, int, str]:
    """Return exactly one function definition, bounded by its braces.

    Do not use the next ``\nvoid `` as an end marker: upstream can place static
    data, helper declarations, or preprocessor blocks between functions.  In
    particular, a post-q6_K global ``#if ... __AVX2__`` block made the old
    chunker accidentally absorb a second AVX2 directive and reject a valid
    function as ambiguous.
    """
    p = t.find(signature)
    if p < 0:
        die('missing function signature: ' + signature)
    open_brace = t.find('{', p + len(signature))
    if open_brace < 0:
        die('missing function body: ' + signature)
    # Guard against accidentally matching a prototype/declaration whose semicolon
    # occurs before the first opening brace.
    semi = t.find(';', p + len(signature), open_brace)
    if semi >= 0:
        die('function signature resolved to a declaration, not a definition: ' + signature)
    close_brace = find_matching_brace(t, open_brace, signature)
    e = close_brace + 1
    return p, e, t[p:e]


def find_matching_brace(s: str, open_pos: int, label: str) -> int:
    depth = 0
    i = open_pos
    in_str = False
    in_chr = False
    esc = False
    while i < len(s):
        c = s[i]
        if esc:
            esc = False
        elif c == '\\' and (in_str or in_chr):
            esc = True
        elif c == '"' and not in_chr:
            in_str = not in_str
        elif c == "'" and not in_str:
            in_chr = not in_chr
        elif not in_str and not in_chr:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    die(f'{label}: unmatched brace')
    raise AssertionError


def find_preprocessor_branch(ch: str, feature: str, tag: str) -> tuple[int, int]:
    """Return the body span of the unique #if/#elif branch for *feature*.

    A function can contain the same semantic loop in its AVX2 and scalar fallback
    branches. Matching the whole function is therefore ambiguous. This lightweight
    preprocessor scanner follows nested #if/#endif depth so harmless fallback or
    nested feature blocks cannot confuse the KNL hot-loop patch.
    """
    feat = re.escape(feature)
    rg = re.compile(
        rf'(?m)^[ \t]*#\s*(?:if|elif)\b[^\n]*(?:defined\s*(?:\(\s*)?{feat}(?:\s*\))?|\b{feat}\b)[^\n]*$'
    )
    ms = list(rg.finditer(ch))
    if len(ms) != 1:
        die(f'{tag}: expected one preprocessor branch for {feature}, found {len(ms)}')
    m = ms[0]
    start = ch.find('\n', m.end())
    start = len(ch) if start < 0 else start + 1

    directive = re.compile(r'(?m)^[ \t]*#\s*(?P<kind>if|ifdef|ifndef|elif|else|endif)\b[^\n]*$')
    depth = 1
    for d in directive.finditer(ch, start):
        kind = d.group('kind')
        if kind in ('if', 'ifdef', 'ifndef'):
            depth += 1
        elif kind == 'endif':
            depth -= 1
            if depth == 0:
                return start, d.start()
        elif kind in ('elif', 'else') and depth == 1:
            return start, d.start()
    die(f'{tag}: unterminated preprocessor branch for {feature}')
    raise AssertionError


def patch_simple_hot_loop(t: str, signature: str, tag: str) -> str:
    p, e, ch = find_function_chunk(t, signature)
    # Patch only the AVX2 implementation. Current x86 K-quant functions also carry
    # a scalar/fallback loop with the same `for (int i = 0; i < nb; ++i)` shape.
    # KNL executes the AVX2 branch, so function-wide matching is both ambiguous and
    # semantically wrong.
    b0, b1 = find_preprocessor_branch(ch, '__AVX2__', tag)
    branch = ch[b0:b1]
    rg = re.compile(r'(?m)^(?P<indent>[ \t]*)for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*nb\s*;\s*\+\+i\s*\)\s*\{')
    ms = list(rg.finditer(branch))
    if len(ms) != 1:
        die(f'{tag}: expected one AVX2 hot-loop anchor, found {len(ms)}')
    m = ms[0]
    # A final semantic guard: this must be the model-block loop, not an unrelated
    # loop that happens to use i/nb. Its body must reference both x[i] and y[i].
    open_brace = branch.find('{', m.start(), m.end() + 1)
    close_brace = find_matching_brace(branch, open_brace, f'{tag} AVX2 hot loop')
    body = branch[m.end():close_brace]
    if 'x[i]' not in body or 'y[i]' not in body:
        die(f'{tag}: AVX2 loop does not reference both x[i] and y[i]; upstream changed')
    indent = m.group('indent')
    repl = (
        f'{indent}GGML_KNL_TRACE_ONCE("{tag}");\n\n'
        f'{m.group(0)}\n'
        f'{indent}    GGML_KNL_PREFETCH_PAIR(x, y, i, nb);'
    )
    branch = branch[:m.start()] + repl + branch[m.end():]
    ch = ch[:b0] + branch + ch[b1:]
    return t[:p] + ch + t[e:]


def patch_q4_0(t: str) -> str:
    sig = 'void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s'
    p, e, ch = find_function_chunk(t, sig)

    # Restrict ourselves to the AVX2 section. This is deliberately semantic rather
    # than byte-for-byte: comments/blank lines and clang-format spacing can change.
    avx = ch.find('#if defined(__AVX2__)')
    if avx < 0:
        die('q4_0: AVX2 section not found')
    terms = [x for x in (ch.find('#elif', avx), ch.find('#else', avx), ch.find('#endif', avx)) if x >= 0]
    if not terms:
        die('q4_0: AVX2 section terminator not found')
    avx_end = min(terms)
    sec = ch[avx:avx_end]

    acc_m = one_match(sec, r'(?m)^(?P<indent>[ \t]*)__m256\s+acc\s*=\s*_mm256_setzero_ps\(\)\s*;\s*$', 'q4_0 accumulator')
    loop_m = one_match(sec, r'(?m)^(?P<indent>[ \t]*)for\s*\(\s*;\s*ib\s*<\s*nb\s*;\s*\+\+ib\s*\)\s*\{', 'q4_0 AVX2 loop')
    if loop_m.start() <= acc_m.end():
        die('q4_0: loop precedes accumulator unexpectedly')
    open_brace = sec.find('{', loop_m.start(), loop_m.end() + 1)
    close_brace = find_matching_brace(sec, open_brace, 'q4_0 AVX2 loop')
    sum_m = one_match(sec[close_brace + 1:], r'(?m)^(?P<indent>[ \t]*)sumf\s*=\s*hsum_float_8\(acc\)\s*;\s*$', 'q4_0 hsum')
    sum_start = close_brace + 1 + sum_m.start()
    sum_end = close_brace + 1 + sum_m.end()

    body = sec[loop_m.end():close_brace]
    # The macro body is the existing upstream computation with only its block index
    # and accumulator abstracted. If upstream changes that computation, this
    # replacement fails loudly rather than silently compiling the wrong math.
    step = body.replace('[ib]', '[IDX]')
    acc_pat = re.compile(r'acc\s*=\s*_mm256_fmadd_ps\(\s*d\s*,\s*q\s*,\s*acc\s*\)\s*;')
    if len(acc_pat.findall(step)) != 1:
        die('q4_0: expected exactly one AVX2 FMA accumulation')
    step = acc_pat.sub('ACCUM = _mm256_fmadd_ps(d, q, ACCUM);', step, count=1)

    macro_lines = []
    for ln in step.splitlines():
        # A // comment is unsafe in a continued macro: backslash-newline splicing
        # happens before comment removal, so it can comment out the following
        # replacement-list lines.  Strip // comments lexically (outside string /
        # character literals); block comments are safe to preserve.
        out = []
        in_str = False
        in_chr = False
        esc = False
        i = 0
        while i < len(ln):
            c = ln[i]
            if esc:
                out.append(c)
                esc = False
                i += 1
                continue
            if c == '\\' and (in_str or in_chr):
                out.append(c)
                esc = True
                i += 1
                continue
            if c == '"' and not in_chr:
                in_str = not in_str
                out.append(c)
                i += 1
                continue
            if c == "'" and not in_str:
                in_chr = not in_chr
                out.append(c)
                i += 1
                continue
            if not in_str and not in_chr and c == '/' and i + 1 < len(ln) and ln[i + 1] == '/':
                break
            out.append(c)
            i += 1
        stripped = ''.join(out).strip()
        if not stripped:
            continue
        if stripped.startswith('#'):
            die('q4_0: preprocessor directive inside AVX2 loop cannot be macro-unrolled safely')
        if stripped.endswith('\\'):
            die('q4_0: source line already ends in backslash; cannot macro-unroll safely')
        macro_lines.append('        ' + stripped + ' \\')
    step_macro = '\n'.join(macro_lines)

    indent = acc_m.group('indent')
    old_prefix = sec[acc_m.start():loop_m.start()]
    # Keep any upstream comments between accumulator declaration and loop only in
    # the non-unrolled path unnecessary; the generated comment explains the split.
    q4new = (
        f'{indent}__m256 acc = _mm256_setzero_ps();\n'
        f'{indent}GGML_KNL_TRACE_ONCE("q4_0_q8_0");\n\n'
        '#if defined(GGML_AVX512_KNL) && defined(GGML_KNL_Q4_0_UNROLL)\n'
        '#define GGML_KNL_Q4_STEP(IDX, ACCUM) do { \\\n' + step_macro + '\n'
        '    } while (0)\n'
        f'{indent}__m256 acc0 = _mm256_setzero_ps();\n'
        f'{indent}__m256 acc1 = _mm256_setzero_ps();\n'
        f'{indent}__m256 acc2 = _mm256_setzero_ps();\n'
        f'{indent}__m256 acc3 = _mm256_setzero_ps();\n'
        f'{indent}for (; ib + 3 < nb; ib += 4) {{\n'
        f'{indent}    GGML_KNL_PREFETCH_PAIR(x, y, ib, nb);\n'
        f'{indent}    GGML_KNL_Q4_STEP(ib + 0, acc0);\n'
        f'{indent}    GGML_KNL_Q4_STEP(ib + 1, acc1);\n'
        f'{indent}    GGML_KNL_Q4_STEP(ib + 2, acc2);\n'
        f'{indent}    GGML_KNL_Q4_STEP(ib + 3, acc3);\n'
        f'{indent}}}\n'
        f'{indent}acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));\n'
        f'{indent}for (; ib < nb; ++ib) {{\n'
        f'{indent}    GGML_KNL_PREFETCH_PAIR(x, y, ib, nb);\n'
        f'{indent}    GGML_KNL_Q4_STEP(ib, acc);\n'
        f'{indent}}}\n'
        '#undef GGML_KNL_Q4_STEP\n'
        '#else\n'
        f'{indent}for (; ib < nb; ++ib) {{\n'
        f'{indent}    GGML_KNL_PREFETCH_PAIR(x, y, ib, nb);' + body +
        f'{indent}}}\n'
        '#endif\n'
        f'{indent}sumf = hsum_float_8(acc);\n'
    )

    # Replace accumulator through hsum, leaving #if/#elif structure untouched.
    sec = sec[:acc_m.start()] + q4new + sec[sum_end:]
    ch = ch[:avx] + sec + ch[avx_end:]
    return t[:p] + ch + t[e:]


def patch_public_options(t: str) -> str:
    m = one_match(
        t,
        r'(?m)^(?P<indent>[ \t]*)option\(\s*GGML_AVX512\b[^\n]*\)\s*$',
        'public options',
    )
    indent = m.group('indent')
    lines = [
        f'# {MARK}: KNL has AVX-512F/CD/ER/PF but not BW/DQ/VL/VNNI. patch={PATCH_REV}',
        'option(GGML_AVX512_KNL "ggml: build Knights Landing AVX-512F/CD/ER/PF CPU variant" OFF)',
        'option(GGML_KNL_PREFETCH "ggml: enable KNL quant software prefetch" ON)',
        'set(GGML_KNL_PREFETCH_DISTANCE "2" CACHE STRING "ggml: KNL quant prefetch distance in blocks")',
        'set(GGML_KNL_PREFETCH_HINT "0" CACHE STRING "ggml: KNL prefetch 0=T0 1=T1 2=NTA")',
        'option(GGML_KNL_Q4_0_UNROLL "ggml: enable KNL 4-way q4_0 vec-dot unroll" ON)',
        'option(GGML_KNL_INSTRUMENT "ggml: compile KNL trace hooks (GGML_KNL_TRACE=1)" OFF)',
    ]
    payload = ''.join(indent + x + '\n' for x in lines)
    return insert_after_match(t, m, payload)


def patch_variant_list(t: str) -> str:
    # Add KNL only to the reset list inside ggml_add_cpu_backend_variant(). The
    # feature names appear in several other CMake lists, so global adjacency is
    # intentionally not used.
    fsig = 'function(ggml_add_cpu_backend_variant tag_name)'
    fp = t.find(fsig)
    if fp < 0:
        die('feature reset: ggml_add_cpu_backend_variant() not found')
    fe = t.find('endfunction()', fp)
    if fe < 0:
        die('feature reset: endfunction() not found')
    chunk = t[fp:fe]
    rg = re.compile(r'\bAVX512(?P<ws>[ \t]+)AVX512_VBMI\b')
    ms = list(rg.finditer(chunk))
    if len(ms) != 1:
        die(f'feature reset: expected one AVX512 -> AVX512_VBMI anchor in variant function, found {len(ms)}')
    m0 = ms[0]
    chunk = chunk[:m0.start()] + 'AVX512' + m0.group('ws') + 'AVX512_KNL' + m0.group('ws') + 'AVX512_VBMI' + chunk[m0.end():]
    t = t[:fp] + chunk + t[fe:]

    m = one_match(
        t,
        r'(?m)^(?P<indent>[ \t]*)ggml_add_cpu_backend_variant\(\s*haswell\b[^\n]*\)\s*$',
        'dynamic variant',
    )
    indent = m.group('indent')
    payload = (
        f'{indent}# {MARK}: GCC 14 is the last GCC with KNL target support.\n'
        f'{indent}if (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 15)\n'
        f'{indent}    ggml_add_cpu_backend_variant(knl SSE42 AVX F16C FMA AVX2 BMI2 AVX512_KNL)\n'
        f'{indent}endif()\n'
    )
    return insert_after_match(t, m, payload)


def patch_compiler_flags(t: str) -> str:
    # There is also an MSVC GGML_AVX512 block. Select the one that actually emits
    # -mavx512f/-mavx512cd, making this resilient to indentation changes.
    starts = list(re.finditer(r'(?m)^(?P<indent>[ \t]*)if\s*\(\s*GGML_AVX512\s*\)\s*$', t))
    candidates = []
    for m in starts:
        tail = t[m.end():m.end() + 900]
        if '-mavx512f' in tail and '-mavx512cd' in tail:
            candidates.append(m)
    if len(candidates) != 1:
        die(f'compiler flags: expected one GNU/Clang AVX512 block, found {len(candidates)}')
    m = candidates[0]
    indent = m.group('indent')
    payload = (
        f'{indent}# {MARK}: KNL-specific target; never reuse generic GGML_AVX512 (BW/DQ/VL).\n'
        f'{indent}if (GGML_AVX512_KNL)\n'
        f'{indent}    if (NOT CMAKE_C_COMPILER_ID STREQUAL "GNU" OR NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 15)\n'
        f'{indent}        message(FATAL_ERROR "GGML_AVX512_KNL requires GCC <= 14")\n'
        f'{indent}    endif()\n'
        f'{indent}    list(APPEND ARCH_FLAGS -march=knl -mtune=knl)\n'
        f'{indent}    list(APPEND ARCH_DEFINITIONS GGML_AVX512_KNL)\n'
        f'{indent}    if (GGML_KNL_PREFETCH)\n'
        f'{indent}        list(APPEND ARCH_DEFINITIONS GGML_KNL_PREFETCH GGML_KNL_PREFETCH_DISTANCE=${{GGML_KNL_PREFETCH_DISTANCE}} GGML_KNL_PREFETCH_HINT=${{GGML_KNL_PREFETCH_HINT}})\n'
        f'{indent}    endif()\n'
        f'{indent}    if (GGML_KNL_Q4_0_UNROLL)\n'
        f'{indent}        list(APPEND ARCH_DEFINITIONS GGML_KNL_Q4_0_UNROLL)\n'
        f'{indent}    endif()\n'
        f'{indent}    if (GGML_KNL_INSTRUMENT)\n'
        f'{indent}        list(APPEND ARCH_DEFINITIONS GGML_KNL_INSTRUMENT)\n'
        f'{indent}    endif()\n'
        f'{indent}endif()\n\n'
    )
    return insert_before_match(t, m, payload)


def patch_cpu_feats(t: str) -> str:
    m = one_match(t, r'(?m)^#ifdef[ \t]+GGML_AVX512[ \t]*$', 'runtime score')
    payload = (
        '#ifdef GGML_AVX512_KNL\n'
        f'    // {MARK}: PF+ER identifies the KNL-era AVX-512 subset; PREFETCHWT1 is required.\n'
        '    if (!is.AVX512F() || !is.AVX512CD() || !is.AVX512PF() || !is.AVX512ER() || !is.PREFETCHWT1()) { return 0; }\n'
        '    score += 1<<7;\n'
        '#endif\n\n'
    )
    return insert_before_match(t, m, payload)


def patch_quants(t: str) -> str:
    m = one_match(t, r'(?m)^#define[ \t]+UNUSED[ \t]+GGML_UNUSED[ \t]*$', 'quant helpers')
    helper = r'''
// KNL_LLAMA_LAB_V3: opt-in KNL hot-loop prefetch / instrumentation.
#if defined(GGML_AVX512_KNL)
#ifndef GGML_KNL_PREFETCH_DISTANCE
#define GGML_KNL_PREFETCH_DISTANCE 2
#endif
#ifndef GGML_KNL_PREFETCH_HINT
#define GGML_KNL_PREFETCH_HINT 0
#endif
#if GGML_KNL_PREFETCH_HINT == 1
#define GGML_KNL_MM_HINT _MM_HINT_T1
#elif GGML_KNL_PREFETCH_HINT == 2
#define GGML_KNL_MM_HINT _MM_HINT_NTA
#else
#define GGML_KNL_MM_HINT _MM_HINT_T0
#endif
#if defined(GGML_KNL_PREFETCH)
#define GGML_KNL_PREFETCH_PAIR(X,Y,I,NB) do { \
    const int ggml_knl_pf_i = (I) + GGML_KNL_PREFETCH_DISTANCE; \
    if (ggml_knl_pf_i < (NB)) { \
        _mm_prefetch((const char *) &(X)[ggml_knl_pf_i], GGML_KNL_MM_HINT); \
        _mm_prefetch((const char *) &(Y)[ggml_knl_pf_i], GGML_KNL_MM_HINT); \
    } \
} while (0)
#else
#define GGML_KNL_PREFETCH_PAIR(X,Y,I,NB) do { (void)(X); (void)(Y); (void)(I); (void)(NB); } while (0)
#endif
#if defined(GGML_KNL_PREFETCH)
#define GGML_KNL_TRACE_PF 1
#else
#define GGML_KNL_TRACE_PF 0
#endif
#if defined(GGML_KNL_Q4_0_UNROLL)
#define GGML_KNL_TRACE_UNROLL 1
#else
#define GGML_KNL_TRACE_UNROLL 0
#endif
#if defined(GGML_KNL_INSTRUMENT)
static int ggml_knl_trace_enabled(void) {
    const char * v = getenv("GGML_KNL_TRACE");
    return v && v[0] && strcmp(v, "0") != 0;
}
#define GGML_KNL_TRACE_ONCE(TAG) do { \
    static int ggml_knl_trace_once = 0; \
    if (!ggml_knl_trace_once && ggml_knl_trace_enabled()) { \
        ggml_knl_trace_once = 1; \
        fprintf(stderr, "[ggml-knl] active vec-dot: %s prefetch=%d pf_dist=%d pf_hint=%d q4_unroll=%d\n", \
            (TAG), GGML_KNL_TRACE_PF, GGML_KNL_PREFETCH_DISTANCE, GGML_KNL_PREFETCH_HINT, GGML_KNL_TRACE_UNROLL); \
    } \
} while (0)
#else
#define GGML_KNL_TRACE_ONCE(TAG) do { (void)(TAG); } while (0)
#endif
#else
#define GGML_KNL_PREFETCH_PAIR(X,Y,I,NB) do { (void)(X); (void)(Y); (void)(I); (void)(NB); } while (0)
#define GGML_KNL_TRACE_ONCE(TAG) do { (void)(TAG); } while (0)
#endif
'''
    t = insert_after_match(t, m, helper)
    t = patch_q4_0(t)
    t = patch_simple_hot_loop(t, 'void ggml_vec_dot_q3_K_q8_K(int n,', 'q3_K_q8_K')
    t = patch_simple_hot_loop(t, 'void ggml_vec_dot_q4_K_q8_K(int n,', 'q4_K_q8_K')
    t = patch_simple_hot_loop(t, 'void ggml_vec_dot_q6_K_q8_K(int n,', 'q6_K_q8_K')
    return t


def transform(files: dict[Path, str]) -> dict[Path, str]:
    out = dict(files)
    out[FILES[0]] = patch_public_options(out[FILES[0]])
    out[FILES[1]] = patch_variant_list(out[FILES[1]])
    out[FILES[2]] = patch_compiler_flags(out[FILES[2]])
    out[FILES[3]] = patch_cpu_feats(out[FILES[3]])
    out[FILES[4]] = patch_quants(out[FILES[4]])
    for f in FILES:
        if MARK not in out[f]:
            die(f'internal validation failed: marker absent from {f}')
    return out


def atomic_write(path: Path, text: str):
    tmp = path.with_name(path.name + '.knl-patch-tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def apply(root: Path):
    for f in FILES:
        if not (root / f).is_file():
            die(f'missing {f}; not current llama.cpp tree?')

    original = {f: (root / f).read_text() for f in FILES}
    state = [MARK in original[f] for f in FILES]
    if all(state):
        print('KNL patch already present; no changes made')
        return
    if any(state):
        patched = ', '.join(str(f) for f, yes in zip(FILES, state) if yes)
        die('partial KNL patch detected in: ' + patched + '; run --revert before re-applying')

    # Transactional: all semantic anchors and source transformations must succeed
    # before the first upstream file is modified.
    patched = transform(original)

    b = root / '.knl-llama-lab-v3-backup'
    b.mkdir(exist_ok=True)
    for f in FILES:
        d = b / f
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            shutil.copy2(root / f, d)
    (b / 'manifest.json').write_text(json.dumps([str(x) for x in FILES], indent=2) + '\n')
    (b / 'patch-revision.txt').write_text(PATCH_REV + '\n')

    for f in FILES:
        atomic_write(root / f, patched[f])
    print('KNL patch applied transactionally; backups at', b)


def dry_run(root: Path):
    for f in FILES:
        if not (root / f).is_file():
            die(f'missing {f}; not current llama.cpp tree?')
    original = {f: (root / f).read_text() for f in FILES}
    state = [MARK in original[f] for f in FILES]
    if all(state):
        print('KNL patch compatibility: already patched')
        return
    if any(state):
        die('KNL patch compatibility: partial patch state; run --revert first')
    transform(original)
    print('KNL patch compatibility: PASS (all semantic anchors resolved; source unchanged)')


def check(root: Path):
    bad = 0
    states = []
    for f in FILES:
        p = root / f
        if not p.exists():
            print('MISSING', f)
            bad = 1
            states.append(False)
        else:
            yes = MARK in p.read_text(errors='replace')
            states.append(yes)
            print(('PATCHED' if yes else 'UNPATCHED'), f)
    if any(states) and not all(states):
        print('ERROR: partial patch state', file=sys.stderr)
        bad = 1
    raise SystemExit(bad)


def revert(root: Path):
    b = root / '.knl-llama-lab-v3-backup'
    mf = b / 'manifest.json'
    if not mf.exists():
        die('no backup manifest')
    for s in json.loads(mf.read_text()):
        src = b / s
        dst = root / s
        if not src.is_file():
            die(f'backup missing {s}')
        shutil.copy2(src, dst)
        print('RESTORED', s)


def main():
    ap = argparse.ArgumentParser(description='Apply/revert the llama.cpp KNL backend patch using semantic, fail-closed anchors.')
    ap.add_argument('root', type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--apply', action='store_true')
    g.add_argument('--dry-run', action='store_true', help='validate all semantic anchors without modifying source')
    g.add_argument('--check', action='store_true')
    g.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    r = a.root.resolve()
    if a.apply:
        apply(r)
    elif a.dry_run:
        dry_run(r)
    elif a.check:
        check(r)
    else:
        revert(r)


if __name__ == '__main__':
    main()
