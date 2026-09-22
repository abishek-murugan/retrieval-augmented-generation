"""Compatibility shims so RAGAS 0.4.x imports cleanly on langchain-community 0.4.x.

ragas/llms/base.py hard-imports `langchain_community.chat_models.vertexai.ChatVertexAI`,
which no longer exists in langchain-community >= 0.3. We inject a stub module into
sys.modules before the first ragas import, plus a `VertexAI` attribute on
`langchain_community.llms`.
"""

from __future__ import annotations

import sys
import types

_installed = False


def _build_vertexai_stub() -> types.ModuleType:
    module = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # noqa: D401 - stub
        def __init__(self, *args, **kwargs): ...

    module.ChatVertexAI = ChatVertexAI
    return module


def install_ragas_compat() -> None:
    global _installed
    if _installed:
        return

    import langchain_community.chat_models
    import langchain_community.llms

    if not hasattr(langchain_community.llms, "VertexAI"):
        langchain_community.llms.VertexAI = type("VertexAI", (), {})  # type: ignore[attr-defined]

    if "langchain_community.chat_models.vertexai" not in sys.modules:
        sys.modules["langchain_community.chat_models.vertexai"] = _build_vertexai_stub()

    _installed = True


def import_ragas():
    """Import the real ragas package after installing compat stubs."""
    install_ragas_compat()
    import ragas  # noqa: PLC0415

    return ragas