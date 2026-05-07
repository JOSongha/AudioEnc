"""Helpers for nubes-direct manifest builders.

All audio_asr builders (MLS / VoxPopuli / GigaSpeech / LibriTTS-R) read from
nubes only — no /mnt/ddn/users/<person>/ and no /mnt/ddn/omni_dataset/...
caches. Nubes is the single source of truth for both audio file enumeration
and transcripts.

The nubes HTTP gateway accepts GET requests of two shapes:

    fetch:  /v1/<bucket>/<object-path>
    list:   /v1/<bucket>?dir=/<path>/&max-contents=<n>[&continuation-token=<tok>]

list returns a JSON array of {Name, Size, IsDir, ...} entries plus an
X-Continuation-Token header when more entries are available. max-contents
caps somewhere between 1000 and 5000 (cap=1000 is safe).
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Iterator

NUBES_GATEWAY = "http://c.nubes.sto.navercorp.com:8000/v1"
BUCKET = "hyperscaleai-audiollm"
LIST_CAP = 1000


def _request(path: str, *, timeout: float = 30.0,
             return_headers: bool = False) -> bytes | tuple[bytes, dict[str, str]]:
    """GET <gateway>/<bucket>/<path> and return body (and optionally headers)."""
    url = f"{NUBES_GATEWAY}/{BUCKET}/{path.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        body = r.read()
        if return_headers:
            headers = {k: v for k, v in r.headers.items()}
            return body, headers
    return body


def fetch_object(nubes_path: str, *, timeout: float = 30.0) -> bytes:
    """Fetch a single nubes object. nubes_path = `<bucket>/<path>` or just `<path>`."""
    if nubes_path.startswith(BUCKET + "/"):
        nubes_path = nubes_path[len(BUCKET) + 1:]
    return _request(nubes_path, timeout=timeout)  # type: ignore[return-value]


def list_dir(prefix: str, *, max_contents: int = LIST_CAP) -> Iterator[dict]:
    """Yield objects under <bucket>/<prefix>/ with continuation-token pagination.

    prefix should end with '/'. Each yielded entry is the raw nubes JSON dict
    {Name, Size, IsDir, ETag, ModTime}. Caller filters by IsDir / extension.
    """
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    token = None
    while True:
        params = {"dir": prefix, "max-contents": str(max_contents)}
        if token:
            params["continuation-token"] = token
        url = (f"{NUBES_GATEWAY}/{BUCKET}?"
               f"{urllib.parse.urlencode(params)}")
        with urllib.request.urlopen(url, timeout=60.0) as r:
            body = r.read()
            headers = r.headers
        try:
            entries = json.loads(body)
        except json.JSONDecodeError:
            entries = []
        if not entries:
            return
        for e in entries:
            yield e
        token = headers.get("X-Continuation-Token")
        if not token:
            return


def list_files_recursive(prefix: str, *, suffix: str | None = None,
                          max_contents: int = LIST_CAP) -> Iterator[str]:
    """Recursively yield object Names (relative to bucket root) under prefix.

    Optionally filter by `suffix` (e.g. '.flac' to skip .txt / metadata).
    Output paths include the leading prefix so the caller has the full
    nubes-relative path ready to feed to fetch_object.
    """
    stack = [prefix.lstrip("/").rstrip("/") + "/"]
    while stack:
        cur = stack.pop()
        for e in list_dir(cur, max_contents=max_contents):
            full = cur + e["Name"]
            if e.get("IsDir"):
                stack.append(full + "/")
                continue
            if suffix is not None and not full.endswith(suffix):
                continue
            yield full


def fetch_text_files_parallel(nubes_paths: Iterable[str], *,
                              workers: int = 64,
                              timeout: float = 15.0) -> Iterator[tuple[str, str]]:
    """Yield (nubes_path, text_content) for each .txt path. Order not preserved.

    Caller is responsible for matching the .txt nubes_path back to the .flac
    nubes_path (typically same stem with extension swapped)."""
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(fetch_object, p, timeout=timeout): p
               for p in nubes_paths}
    try:
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                body = fut.result()
            except Exception as e:
                yield (p, "")  # caller can detect empty + log
                continue
            text = body.decode("utf-8", errors="replace").strip()
            yield (p, text)
    finally:
        # cancel_futures=True is Py3.9+; fall back to plain shutdown for Py3.8.
        pool.shutdown(wait=False)


def flac_to_txt_path(flac_path: str) -> str:
    """Swap .flac suffix → .txt (per-file transcript convention used by
    GigaSpeech and VoxPopuli on nubes)."""
    assert flac_path.endswith(".flac"), flac_path
    return flac_path[:-5] + ".txt"
