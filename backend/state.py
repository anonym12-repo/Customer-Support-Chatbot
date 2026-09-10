"""Conversation state management.

Structured state assists the LLM instead of forcing it to infer everything from
raw chat history (e.g. resolving "it" to the previously mentioned order).
"""
from __future__ import annotations

import uuid

from .schemas import ConversationState, RouterResult

# In-memory store keyed by conversation id. In production this would be a
# persistent store (Redis / DB); for the demo a dict is sufficient.
_STORE: dict[str, ConversationState] = {}


def new_conversation() -> ConversationState:
    state = ConversationState(conversation_id=str(uuid.uuid4()))
    _STORE[state.conversation_id] = state
    return state


def get_state(conversation_id: str) -> ConversationState:
    state = _STORE.get(conversation_id)
    if state is None:
        state = ConversationState(conversation_id=conversation_id)
        _STORE[conversation_id] = state
    return state


def update_state(state: ConversationState, router: RouterResult, answer: str) -> None:
    state.update_from_router(router)
    state.history.append({"role": "user", "content": router.raw.get("text", "")})
    state.history.append({"role": "assistant", "content": answer})


def reset_store() -> None:
    """Clear all conversations (used by tests)."""
    _STORE.clear()