"""Unified LLM client: provider routing, per-provider rate limiting and concurrency, token budgeting, and JSON-mode calls."""
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled by get_*_client functions
    OpenAI = None

from src.common import approx_token_count, budget_messages, extract_json
from src.config import ENV_PATH

_LOADED_ENV_PATHS: List[str] = []
try:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)
        _LOADED_ENV_PATHS.append(str(ENV_PATH))
except Exception:
    # Do not crash at import time. Missing keys are reported when a client is used.
    pass


DEFAULT_MAX_RETRIES = 5
DEFAULT_SLEEP_SECONDS = 2

# ---------------------------------------------------------------------------
# Global LLM rate limiter (v62).  When many families run concurrently, the most
# efficient throttle is a per-provider concurrency cap plus an optional
# requests-per-minute token bucket — this lets us push as many parallel requests
# as the provider tier allows without tripping HTTP 429, instead of the old
# fixed per-module worker pools.  Both limits are shared across ALL threads.
#
#   LLM_MAX_CONCURRENCY   max in-flight requests per provider (default 8)
#   LLM_MAX_RPM           requests/minute per provider, 0 = unlimited (default 0)
# Per-provider overrides: LLM_MAX_CONCURRENCY_OPENROUTER, LLM_MAX_RPM_DEEPSEEK, …
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class _ProviderLimiter:
    """Token-bucket (RPM) + bounded concurrency for one provider, thread-safe."""

    def __init__(self, rpm: int, max_concurrency: int):
        self.rpm = max(0, rpm)
        self.capacity = float(self.rpm)
        self.tokens = float(self.rpm)
        self.refill_per_sec = self.rpm / 60.0 if self.rpm > 0 else 0.0
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.sem = threading.BoundedSemaphore(max_concurrency) if max_concurrency > 0 else None

    def _take_token(self) -> None:
        if self.rpm <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.last) * self.refill_per_sec
                )
                self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.refill_per_sec if self.refill_per_sec else 0.5
            time.sleep(min(max(wait, 0.01), 1.0))


_LIMITERS: Dict[str, _ProviderLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def _get_limiter(provider: str) -> _ProviderLimiter:
    provider = (provider or "default").lower()
    with _LIMITERS_LOCK:
        lim = _LIMITERS.get(provider)
        if lim is None:
            suffix = provider.upper()
            conc = _env_int(f"LLM_MAX_CONCURRENCY_{suffix}", _env_int("LLM_MAX_CONCURRENCY", 8))
            rpm = _env_int(f"LLM_MAX_RPM_{suffix}", _env_int("LLM_MAX_RPM", 0))
            lim = _ProviderLimiter(rpm=rpm, max_concurrency=conc)
            _LIMITERS[provider] = lim
        return lim


@contextmanager
def llm_slot(provider: str):
    """Hold one concurrency slot for the network call and pace by RPM token bucket."""
    lim = _get_limiter(provider)
    if lim.sem is not None:
        lim.sem.acquire()
    try:
        lim._take_token()
        yield
    finally:
        if lim.sem is not None:
            lim.sem.release()


# Provider defaults: generation uses OpenRouter; evaluation/Judge uses DeepSeek.
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL_NAME", os.getenv("OPENROUTER_GENERATION_MODEL", "openai/gpt-4o-mini")
)
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

_DEEPSEEK_CLIENT_CACHE = None
_OPENROUTER_CLIENT_CACHE = None
_OPENAI_CLIENT_CACHE = None


def _missing_key_message(key_name: str, example: str) -> str:
    """Build a user-friendly API-key error message."""
    return f"没有找到 {key_name}。\n" f"请检查 code/.env 文件是否存在：\n{ENV_PATH}\n" f"并写入：\n{example}"


def _get_input_budget() -> int:
    """Read the global input budget from .env with a safe fallback."""
    try:
        return int(os.getenv("LLM_INPUT_TOKEN_BUDGET", "6500"))
    except Exception:
        return 6500


def _usage_dict(response: Any) -> Dict[str, Optional[int]]:
    """Normalize provider usage objects into a simple dictionary."""
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }


def _messages_token_estimate(messages: List[Dict[str, str]]) -> int:
    """Approximate tokens for a message list when API usage is unavailable."""
    text = "\n".join(str(m.get("role", "")) + ": " + str(m.get("content", "")) for m in messages)
    return approx_token_count(text)


def _prepare_messages(
    messages: List[Dict[str, str]], max_tokens: int, json_mode: bool
) -> List[Dict[str, str]]:
    """Add output constraints and then apply the global input token budget."""
    limited = add_output_token_limit_instruction(
        messages, max_tokens=max_tokens, json_mode=json_mode
    )
    return budget_messages(limited, max_input_tokens=_get_input_budget())


def _should_log_tokens(finish_reason: str, output_tokens: int, max_tokens: int) -> bool:
    """Decide whether token information should be printed.

    ``warn`` prints only near-limit and truncated calls; ``all`` prints every call;
    ``off`` disables logs.
    """
    mode = os.getenv("LLM_TOKEN_LOG_MODE", "warn").strip().lower()
    if mode in {"off", "false", "0", "none"}:
        return False
    if mode == "all":
        return True
    try:
        warn_ratio = float(os.getenv("LLM_TOKEN_WARN_RATIO", "0.85"))
    except Exception:
        warn_ratio = 0.85
    return finish_reason == "length" or (
        max_tokens > 0 and output_tokens >= int(max_tokens * warn_ratio)
    )


def _log_token_usage(
    provider: str,
    mode: str,
    model: str,
    raw_messages: List[Dict[str, str]],
    final_messages: List[Dict[str, str]],
    response: Any,
    content: str,
    max_tokens: int,
    finish_reason: str,
) -> None:
    """Print compact token diagnostics for debugging long prompts/outputs."""
    usage = _usage_dict(response)
    output_tokens = usage.get("completion_tokens") or approx_token_count(content)
    if not _should_log_tokens(finish_reason, int(output_tokens), int(max_tokens)):
        return

    raw_input_est = _messages_token_estimate(raw_messages)
    final_input_est = _messages_token_estimate(final_messages)
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    status = "TRUNCATED" if finish_reason == "length" else "NEAR_LIMIT"

    print(
        f"[{provider} Token] {mode.upper()} {status}: "
        f"model={model}, finish_reason={finish_reason}, "
        f"input≈{final_input_est} tokens(raw≈{raw_input_est}), "
        f"output={output_tokens}/{max_tokens}, "
        f"api_prompt={prompt_tokens}, api_total={total_tokens}, chars={len(content)}"
    )


def _length_error(provider: str, max_tokens: int, content: str, response: Any) -> ValueError:
    """Build a detailed truncation error that includes output-token count."""
    usage = _usage_dict(response)
    output_tokens = usage.get("completion_tokens") or approx_token_count(content)
    return ValueError(
        f"{provider} 输出被 max_tokens 截断。\n"
        f"当前 max_tokens={max_tokens}。\n"
        f"本次已输出 token 数≈{output_tokens}/{max_tokens}。\n"
        f"返回字符数={len(content)}。\n"
        f"建议：增加 max_tokens，或者继续缩短输出 schema。\n"
        f"返回内容开头: {repr(str(content)[:300])}"
    )


def add_output_token_limit_instruction(
    messages: List[Dict[str, str]], max_tokens: int, json_mode: bool = True
) -> List[Dict[str, str]]:
    """Append a short output-length rule to the last user message.

    This is a soft instruction.  The hard limit is still the API ``max_tokens``.
    """
    try:
        token_limit = int(max_tokens)
    except Exception:
        token_limit = 1000

    safe_limit = max(200, token_limit - 80)
    instruction = (
        "\n\nOutput token limit:\n"
        f"- Your completion must stay under {safe_limit} output tokens.\n"
        "- Use compact JSON and short strings.\n"
        "- Do not add explanations outside JSON.\n"
        "- If space is limited, keep required keys and summarize optional details.\n"
        "- Prefer arrays with only the minimum requested items."
    )
    if json_mode:
        instruction += "\n- Return one valid JSON object only."

    if not messages:
        return [{"role": "user", "content": instruction.strip()}]

    copied = [dict(item) for item in messages]
    target_index = len(copied) - 1
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user":
            target_index = index
            break

    content = str(copied[target_index].get("content", ""))
    if "Output token limit:" not in content:
        copied[target_index]["content"] = content.rstrip() + instruction
    return copied


def get_deepseek_api_key() -> str:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            _missing_key_message("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY=你的DeepSeek API Key")
        )
    return api_key


def get_deepseek_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("请先安装或升级 openai 包：pip install -U openai")
    global _DEEPSEEK_CLIENT_CACHE
    if _DEEPSEEK_CLIENT_CACHE is None:
        _DEEPSEEK_CLIENT_CACHE = OpenAI(api_key=get_deepseek_api_key(), base_url=DEEPSEEK_BASE_URL)
    return _DEEPSEEK_CLIENT_CACHE


def call_deepseek_json(
    messages: List[Dict[str, str]],
    model: str = DEEPSEEK_MODEL_NAME,
    max_tokens: int = 3000,
    temperature: float = 0.3,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_seconds: int = DEFAULT_SLEEP_SECONDS,
) -> Dict[str, Any]:
    """Call DeepSeek and parse a JSON object response."""
    client = get_deepseek_client()
    for attempt in range(max_retries):
        try:
            final_messages = _prepare_messages(messages, max_tokens=max_tokens, json_mode=True)
            with llm_slot("deepseek"):
                response = client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", "")
            content = choice.message.content or ""
            _log_token_usage(
                "DeepSeek",
                "json",
                model,
                messages,
                final_messages,
                response,
                content,
                max_tokens,
                finish_reason,
            )

            if not content.strip():
                raise ValueError("DeepSeek 返回空内容")
            if finish_reason == "length":
                raise _length_error("DeepSeek", max_tokens, content, response)

            return {"data": extract_json(content), "usage": _usage_dict(response)}
        except Exception as exc:
            print(f"DeepSeek 第 {attempt + 1} 次调用失败: {exc}")
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError("DeepSeek 多次调用失败，已超过最大重试次数。")


def get_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            _missing_key_message("OPENROUTER_API_KEY", "OPENROUTER_API_KEY=你的OpenRouter API Key")
        )
    return api_key


def get_openrouter_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("请先安装或升级 openai 包：pip install -U openai")
    global _OPENROUTER_CLIENT_CACHE
    if _OPENROUTER_CLIENT_CACHE is None:
        _OPENROUTER_CLIENT_CACHE = OpenAI(
            api_key=get_openrouter_api_key(), base_url=OPENROUTER_BASE_URL
        )
    return _OPENROUTER_CLIENT_CACHE


def call_openrouter_json(
    messages: List[Dict[str, str]],
    model: str = OPENROUTER_MODEL_NAME,
    max_tokens: int = 3000,
    temperature: float = 0.1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_seconds: int = DEFAULT_SLEEP_SECONDS,
) -> Dict[str, Any]:
    """Call OpenRouter and parse a JSON object response."""
    client = get_openrouter_client()
    for attempt in range(max_retries):
        try:
            final_messages = _prepare_messages(messages, max_tokens=max_tokens, json_mode=True)
            with llm_slot("openrouter"):
                response = client.chat.completions.create(
                    model=model or OPENROUTER_MODEL_NAME,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", "")
            content = choice.message.content or ""
            _log_token_usage(
                "OpenRouter",
                "json",
                model or OPENROUTER_MODEL_NAME,
                messages,
                final_messages,
                response,
                content,
                max_tokens,
                finish_reason,
            )

            if not content.strip():
                raise ValueError("OpenRouter 返回空内容")
            if finish_reason == "length":
                raise _length_error("OpenRouter", max_tokens, content, response)

            return {"data": extract_json(content), "usage": _usage_dict(response)}
        except Exception as exc:
            error_text = str(exc)
            non_retryable = [
                "invalid_api_key",
                "Unauthorized",
                "401",
                "model not found",
                "No endpoints found",
                "response_format",
            ]
            if any(key in error_text for key in non_retryable):
                raise exc
            if attempt >= max_retries - 1:
                raise exc
            print(f"[OpenRouter] 第 {attempt + 1}/{max_retries} 次调用失败: {exc}")
            time.sleep(sleep_seconds)
    raise RuntimeError("OpenRouter 调用失败")


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(_missing_key_message("OPENAI_API_KEY", "OPENAI_API_KEY=你的OpenAI API Key"))
    return api_key


def get_openai_client() -> OpenAI:
    if OpenAI is None:
        raise ImportError("请先安装或升级 openai 包：pip install -U openai")
    global _OPENAI_CLIENT_CACHE
    if _OPENAI_CLIENT_CACHE is None:
        _OPENAI_CLIENT_CACHE = OpenAI(api_key=get_openai_api_key())
    return _OPENAI_CLIENT_CACHE


def call_openai_json(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL_NAME,
    max_tokens: int = 3000,
    temperature: float = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_seconds: int = DEFAULT_SLEEP_SECONDS,
) -> Dict[str, Any]:
    """Call OpenAI and parse a JSON object response.

    This project does not require OpenAI by default, but this fallback is kept so
    experiments can switch providers using .env without editing module code.
    """
    client = get_openai_client()
    for attempt in range(max_retries):
        try:
            final_messages = _prepare_messages(messages, max_tokens=max_tokens, json_mode=True)
            with llm_slot("openai"):
                response = client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", "")
            content = choice.message.content or ""
            _log_token_usage(
                "OpenAI",
                "json",
                model,
                messages,
                final_messages,
                response,
                content,
                max_tokens,
                finish_reason,
            )

            if not content.strip():
                raise ValueError("OpenAI 返回空内容")
            if finish_reason == "length":
                raise _length_error("OpenAI", max_tokens, content, response)

            return {"data": extract_json(content), "usage": _usage_dict(response)}
        except Exception as exc:
            error_text = str(exc)
            non_retryable = [
                "Unsupported parameter",
                "Unsupported value",
                "invalid_request_error",
                "max_tokens",
                "temperature",
            ]
            if any(key in error_text for key in non_retryable):
                raise exc
            if attempt >= max_retries - 1:
                raise exc
            print(f"[OpenAI] 第 {attempt + 1}/{max_retries} 次调用失败: {exc}")
            time.sleep(sleep_seconds)
    raise RuntimeError("OpenAI 调用失败")


def call_generation_json(
    messages: List[Dict[str, str]],
    model: str = "",
    max_tokens: int = 3000,
    temperature: float = 0.3,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_seconds: int = DEFAULT_SLEEP_SECONDS,
) -> Dict[str, Any]:
    """Route generation calls to the provider selected in .env."""
    provider = (
        os.getenv("GENERATION_LLM_PROVIDER", os.getenv("LLM_GENERATION_PROVIDER", "openrouter"))
        .strip()
        .lower()
    )
    if provider == "deepseek":
        generation_model = model or os.getenv("DEEPSEEK_GENERATION_MODEL", DEEPSEEK_MODEL_NAME)
        return call_deepseek_json(
            messages, generation_model, max_tokens, temperature, max_retries, sleep_seconds
        )
    if provider in {"openrouter", "openouter"}:
        generation_model = (
            os.getenv("OPENROUTER_GENERATION_MODEL") or model or OPENROUTER_MODEL_NAME
        )
        return call_openrouter_json(
            messages, generation_model, max_tokens, temperature, max_retries, sleep_seconds
        )
    if provider == "openai":
        generation_model = model or os.getenv("OPENAI_GENERATION_MODEL", OPENAI_MODEL_NAME)
        return call_openai_json(
            messages, generation_model, max_tokens, temperature, max_retries, sleep_seconds
        )
    raise ValueError(f"不支持的 GENERATION_LLM_PROVIDER: {provider}。可选：deepseek / openrouter / openai。")


def call_llm_judge_json(
    messages: List[Dict[str, str]],
    model: str = "",
    max_tokens: int = 1800,
    temperature: float = 0.1,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_seconds: int = DEFAULT_SLEEP_SECONDS,
) -> Dict[str, Any]:
    """Route LLM-judge calls to the provider selected in .env."""
    provider = (
        os.getenv("LLM_JUDGE_PROVIDER", os.getenv("EVALUATION_LLM_PROVIDER", "deepseek"))
        .strip()
        .lower()
    )
    if provider in {"openrouter", "openouter"}:
        judge_model = model or os.getenv("OPENROUTER_LLM_JUDGE_MODEL", OPENROUTER_MODEL_NAME)
        return call_openrouter_json(
            messages, judge_model, max_tokens, temperature, max_retries, sleep_seconds
        )
    if provider == "openai":
        judge_model = model or os.getenv("OPENAI_LLM_JUDGE_MODEL", OPENAI_MODEL_NAME)
        return call_openai_json(
            messages, judge_model, max_tokens, temperature, max_retries, sleep_seconds
        )
    if provider == "deepseek":
        judge_model = (
            os.getenv("DEEPSEEK_LLM_JUDGE_MODEL")
            or model
            or os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
        )
        return call_deepseek_json(
            messages, judge_model, max_tokens, temperature, max_retries, sleep_seconds
        )
    raise ValueError(f"不支持的 LLM_JUDGE_PROVIDER: {provider}。可选：openrouter / openai / deepseek。")
