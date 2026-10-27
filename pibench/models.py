"""Model wrappers. Every model has the same method: generate(messages, seed) -> ModelResponse.

- GeminiModel: Google's hosted API (needs GEMINI_API_KEY in .env)
- OllamaModel: local models served by Ollama (needs `ollama serve` running)
- MockModel: an offline stand-in for TESTING THE PLUMBING ONLY. It is
  deliberately gullible and its outputs are never research results.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import requests

from .types import Messages, ModelResponse


class BaseModel:
    name: str = "base"

    def generate(self, messages: Messages, seed: int = 0) -> ModelResponse:  # pragma: no cover
        raise NotImplementedError

    def cache_identity(self) -> str:
        """Everything that changes the output must be in here, so the cache stays correct."""
        return self.name


class GeminiModel(BaseModel):
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 400,
                 min_interval_s: float = 4.0, retries: int = 5, **_: Any):
        self.model = model
        self.name = f"gemini:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval_s = min_interval_s  # spacing between calls, to stay under free-tier limits
        self.retries = retries
        self._last = 0.0
        self.key = os.environ.get("GEMINI_API_KEY", "")

    def cache_identity(self) -> str:
        return f"{self.name}|t={self.temperature}|max={self.max_tokens}"

    def generate(self, messages: Messages, seed: int = 0) -> ModelResponse:
        if not self.key:
            return ModelResponse(text="", error="GEMINI_API_KEY is not set (put it in .env)")
        body = {
            "systemInstruction": {"parts": [{"text": messages.system}]},
            "contents": [{"role": "user", "parts": [{"text": messages.user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "seed": seed,
            },
        }
        headers = {"x-goog-api-key": self.key, "Content-Type": "application/json"}
        wait, err = 5.0, "no attempt made"
        for attempt in range(self.retries + 1):
            gap = time.time() - self._last
            if gap < self.min_interval_s:
                time.sleep(self.min_interval_s - gap)
            t0 = time.time()
            self._last = t0
            try:
                r = requests.post(self.URL.format(model=self.model), json=body, headers=headers, timeout=90)
            except requests.RequestException as e:
                err = f"network error: {e}"
            else:
                if r.status_code == 200:
                    data = r.json()
                    text = _gemini_text(data)
                    return ModelResponse(text=text, seconds=round(time.time() - t0, 3))
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    return ModelResponse(text="", error=err)
            if attempt < self.retries:
                time.sleep(wait)
                wait = min(wait * 2, 120)
        return ModelResponse(text="", error=err)


def list_gemini_models() -> tuple[list[str], str | None]:
    """Model ids this API key can call with generateContent. Returns (ids, error)."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return [], "GEMINI_API_KEY is not set (put it in .env)"
    ids, token = [], None
    for _ in range(10):  # the list is paged
        params = {"pageSize": 100, **({"pageToken": token} if token else {})}
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models", params=params,
                             headers={"x-goog-api-key": key}, timeout=30)
        except requests.RequestException as e:
            return ids, f"network error: {e}"
        if r.status_code != 200:
            return ids, f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        ids += [m["name"].removeprefix("models/") for m in data.get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])]
        token = data.get("nextPageToken")
        if not token:
            break
    return ids, None


def _gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        # A safety block or an empty answer: record it as an empty answer, not a crash.
        return ""


class OllamaModel(BaseModel):
    def __init__(self, model: str, host: str = "http://localhost:11434", temperature: float = 0.0,
                 max_tokens: int = 400, **_: Any):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def cache_identity(self) -> str:
        return f"{self.name}|t={self.temperature}|max={self.max_tokens}"

    def generate(self, messages: Messages, seed: int = 0) -> ModelResponse:
        body = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": messages.system},
                {"role": "user", "content": messages.user},
            ],
            "options": {"temperature": self.temperature, "seed": seed, "num_predict": self.max_tokens},
        }
        t0 = time.time()
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=300)
        except requests.RequestException as e:
            return ModelResponse(text="", error=f"cannot reach Ollama at {self.host}: {e}")
        if r.status_code != 200:
            return ModelResponse(text="", error=f"HTTP {r.status_code}: {r.text[:300]}")
        text = r.json().get("message", {}).get("content", "").strip()
        return ModelResponse(text=text, seconds=round(time.time() - t0, 3))


class OpenAICompatModel(BaseModel):
    """Any server that speaks the OpenAI chat-completions format: Groq, OpenRouter, Together,
    LM Studio, vLLM and others. Useful if the Gemini free tier runs out."""

    def __init__(self, model: str, base_url: str, api_key_env: str = "OPENAI_API_KEY", label: str = "",
                 temperature: float = 0.0, max_tokens: int = 400, min_interval_s: float = 0.0,
                 retries: int = 5, **_: Any):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.name = f"{label or 'openai'}:{model}"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval_s = min_interval_s
        self.retries = retries
        self._last = 0.0

    def cache_identity(self) -> str:
        return f"{self.name}|{self.base_url}|t={self.temperature}|max={self.max_tokens}"

    def generate(self, messages: Messages, seed: int = 0) -> ModelResponse:
        key = os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": messages.system},
                         {"role": "user", "content": messages.user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": seed,
        }
        wait, err = 5.0, "no attempt made"
        for attempt in range(self.retries + 1):
            gap = time.time() - self._last
            if gap < self.min_interval_s:
                time.sleep(self.min_interval_s - gap)
            t0 = self._last = time.time()
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=120)
            except requests.RequestException as e:
                err = f"network error: {e}"
            else:
                if r.status_code == 200:
                    try:
                        text = r.json()["choices"][0]["message"].get("content") or ""
                    except (KeyError, IndexError, ValueError):
                        text = ""
                    return ModelResponse(text=text.strip(), seconds=round(time.time() - t0, 3))
                err = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    return ModelResponse(text="", error=err)
            if attempt < self.retries:
                time.sleep(wait)
                wait = min(wait * 2, 120)
        return ModelResponse(text="", error=err)


class MockModel(BaseModel):
    """Offline, deterministic, and gullible on purpose. FOR TESTS ONLY.

    It answers with the source sentence that best matches the question (citing that source),
    and it obeys simple injected commands it finds in the prompt. That makes it useful for
    checking that retrieval, attacks, checkers and the runner are wired up correctly. It says
    nothing about how a real model behaves, so never report its numbers.
    """
    name = "mock"

    def __init__(self, **_: Any):
        pass

    def generate(self, messages: Messages, seed: int = 0) -> ModelResponse:
        prompt = messages.user
        out: list[str] = []
        say = re.search(r"(?:reply with|respond with|start your reply with|include|say|print|output)"
                        r"[^\"']{0,40}[\"']([^\"']{3,80})[\"']", prompt, re.I)
        if say:
            out.append(say.group(1))
        site = re.search(r"\b(?:visit|go to|website|recommend)\b[^.]{0,60}?\b([a-z0-9-]+\.example)\b", prompt, re.I)
        if site:
            out.append(f"Please visit {site.group(1)}.")
        if re.search(r"(system prompt|reference code|confidential codes?)", prompt, re.I) and \
                re.search(r"(reveal|repeat|print)", prompt, re.I):
            m = re.search(r"reference code:\s*([A-Z0-9-]+)", messages.system)
            if m:
                out.append(f"The code is {m.group(1)}.")
        tool = re.search(r"(send_email\([^)]*\))", prompt)
        if tool:
            out.append(f"TOOL_CALL: {tool.group(1)}")
        out.append(_best_sentence(prompt))
        return ModelResponse(text="\n".join(out), seconds=0.0)


def _best_sentence(prompt: str) -> str:
    from .index import tokenize

    q = re.search(r"Question:\s*(.+)", prompt)
    qtext = (q.group(1) if q else "").strip()
    qwords = set(tokenize(qtext))
    best, best_score = "I could not find that in the sources.", 0
    source = None
    for line in prompt.splitlines():
        head = re.match(r"\[(\d+)\]", line)
        if head:
            source = head.group(1)
            continue
        if line.strip().lower().startswith("question:"):
            continue
        for s in re.split(r"(?<=[.!?])\s+", line):
            score = len(qwords & set(tokenize(s)))
            if score > best_score:
                best, best_score = s.strip() + (f" [{source}]" if source else ""), score
    return best


def build_model(spec: dict[str, Any]) -> BaseModel:
    """Make a model from one entry of config.yaml's `models` list."""
    spec = dict(spec)
    kind = spec.pop("provider")
    if kind == "gemini":
        return GeminiModel(**spec)
    if kind == "ollama":
        return OllamaModel(**spec)
    if kind == "openai_compat":
        return OpenAICompatModel(**spec)
    if kind == "mock":
        return MockModel(**spec)
    raise ValueError(f"Unknown provider '{kind}'. Use gemini, ollama, openai_compat or mock.")
