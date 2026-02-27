import os
import sys
from dataclasses import dataclass

DEFAULT_LOCAL_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


@dataclass
class OllamaSettings:
    target: str
    base_url: str
    chat_model: str
    embedding_model: str


def get_python_compatibility_message(version_info=None):
    """Return a warning string when Python version is outside tested range."""
    info = version_info or sys.version_info
    major, minor = info.major, info.minor

    if major != 3:
        return f"Python {major}.{minor} detected. This project is tested on Python 3.10-3.11."

    if minor > 11:
        return (
            f"Python {major}.{minor} detected. This project is currently tested on Python 3.10-3.11; "
            "newer versions may fail due to dependency compatibility."
        )

    if minor < 10:
        return f"Python {major}.{minor} detected. This project requires Python 3.10 or newer."

    return ""


def resolve_ollama_settings(target="local", remote_url="", chat_model=None, embedding_model=None):
    """Build centralized Ollama config used by UI and terminal flows."""
    normalized_target = (target or "local").strip().lower()
    local_url = os.getenv("OLLAMA_LOCAL_URL", DEFAULT_LOCAL_OLLAMA_URL)

    if normalized_target == "remote":
        base_url = (remote_url or os.getenv("OLLAMA_REMOTE_URL", "")).strip() or local_url
    else:
        base_url = local_url

    return OllamaSettings(
        target=normalized_target,
        base_url=base_url,
        chat_model=(chat_model or DEFAULT_CHAT_MODEL).strip(),
        embedding_model=(embedding_model or DEFAULT_EMBEDDING_MODEL).strip(),
    )
