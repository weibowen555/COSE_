"""
GPT fallback parser for Validator/Judge score recovery.

When the actor's Validator or Judge rollout fails to emit a parseable
<score>X</score> tag (regex `_extract_scores` returns empty), we re-run
the same prompt template through gpt-4.1-nano (or whichever
BENCH_JUDGE_MODEL is configured) to obtain a fresh score. This recovers
training signal that would otherwise default to 0 (the dominant cause
of judge_score collapse on small/Qwen3-hybrid models).

Design:
- One round-robin OpenAI client pool, fed by api.json (same as benchmark eval).
- Reuses the BENCH_JUDGE_MODEL env var so configuration stays consistent.
- Concurrent fan-out via ThreadPoolExecutor — one request per failed
  parse, collected in parallel.
- Hard timeout per request (10s) so a stuck call can't stall training.
- Returns a normalized float in [0, 1] or None on total failure.

This module is OPTIONAL — the trainer falls back to the legacy zero
default if the helper raises, no api keys are configured, or the env
flag cose.cose.judge_parse_fallback is False.
"""

from __future__ import annotations

import itertools
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from typing import List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


_BENCH_JUDGE_BACKEND = os.environ.get("BENCH_JUDGE_BACKEND", "openai").lower()
_BENCH_JUDGE_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "nim": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}
_BENCH_JUDGE_BASE_URL = (
    os.environ.get("BENCH_JUDGE_BASE_URL")
    or _BENCH_JUDGE_BASE_URLS.get(_BENCH_JUDGE_BACKEND, _BENCH_JUDGE_BASE_URLS["openai"])
)
_BENCH_JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gpt-4.1-nano")


def _parse_score(text: str) -> Optional[float]:
    """Extract a 1-10 score from the GPT response, normalized to [0, 1].

    Mirrors the actor-side `_extract_scores` cascade: try <score>X</score>
    first, fall back to the last numeric in the text. Returns None if
    nothing parseable is found.
    """
    if not text:
        return None
    # <score>X</score> first
    for match in re.finditer(r"<score>(.*?)</score>", text, re.DOTALL):
        chunk = match.group(1).strip()
        for pat in [r"\b(\d+)\s*/\s*(\d+)\b", r"\b(\d+\.\d+)\b", r"\b(\d+)\b"]:
            m = re.search(pat, chunk)
            if m:
                try:
                    if "/" in m.group(0):
                        n, d = int(m.group(1)), int(m.group(2))
                        if d > 0:
                            v = float(Fraction(n, d))
                        else:
                            continue
                    else:
                        v = float(m.group(1))
                    return (max(1.0, min(10.0, v)) - 1.0) / 9.0
                except (ValueError, ZeroDivisionError):
                    continue
    # Fallback: last number anywhere
    nums = re.findall(r"\b\d+/\d+\b|\b\d+\.\d+\b|\b\d+\b", text)
    if nums:
        last = nums[-1]
        try:
            if "/" in last:
                n, d = last.split("/")
                if int(d) > 0:
                    v = float(Fraction(int(n), int(d)))
                else:
                    return None
            else:
                v = float(last)
            return (max(1.0, min(10.0, v)) - 1.0) / 9.0
        except (ValueError, ZeroDivisionError):
            pass
    return None


class GPTScoreFallback:
    """Round-robin GPT score-extractor for Validator/Judge parse failures.

    Usage:
        fallback = GPTScoreFallback(api_keys)
        score = fallback.score_one(prompt_text)              # blocking, single
        scores = fallback.score_many(prompt_texts, max_workers=15)  # parallel
    """

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        model: str = _BENCH_JUDGE_MODEL,
        base_url: str = _BENCH_JUDGE_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 10.0,
    ):
        self.api_keys = api_keys or []
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enabled = bool(self.api_keys) and OpenAI is not None
        if self.enabled:
            self._clients = [
                OpenAI(api_key=k, base_url=self.base_url, timeout=self.timeout)
                for k in self.api_keys
            ]
            self._cycle = itertools.cycle(list(zip(self.api_keys, self._clients)))
            self._lock = threading.Lock()
        else:
            self._clients = []
            self._cycle = None
            self._lock = threading.Lock()

    def _next_client(self) -> Optional[OpenAI]:
        if not self.enabled:
            return None
        with self._lock:
            return next(self._cycle)[1]

    def score_one(self, prompt_text: str) -> Optional[float]:
        """Send `prompt_text` to GPT, parse a score in [0, 1], return None on failure."""
        if not self.enabled:
            return None
        client = self._next_client()
        if client is None:
            return None
        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            text = completion.choices[0].message.content or ""
            return _parse_score(text)
        except Exception:
            return None

    def score_many(
        self,
        prompt_texts: List[str],
        max_workers: int = 15,
    ) -> List[Optional[float]]:
        """Score a batch of prompts in parallel. Returns a list of [0,1] floats
        or None per index. Order matches input."""
        if not self.enabled or not prompt_texts:
            return [None] * len(prompt_texts)
        results: List[Optional[float]] = [None] * len(prompt_texts)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {
                ex.submit(self.score_one, t): i
                for i, t in enumerate(prompt_texts)
            }
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    results[i] = None
        return results
