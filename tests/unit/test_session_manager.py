"""
CredVoice — Unit Tests for SessionManager (tests/unit/test_session_manager.py).

Verifies call session isolation, thread safety, TTL cleanup, idempotency,
and terminal stage protection.
"""

import time
import pytest

from app.agent.llm import StubLLMAdapter
from app.models.conversation import Stage
from app.services.session_manager import CallSession, InMemorySessionStore, SessionManager


def test_session_isolation():
    """Verify separate call_ids maintain completely isolated states."""
    sm = SessionManager(store=InMemorySessionStore())
    s1 = sm.get_or_create_session(call_id="CALL-001", customer_name="Priya")
    s2 = sm.get_or_create_session(call_id="CALL-002", customer_name="Ramesh")

    assert s1.orchestrator.state.session_id == "CALL-001"
    assert s2.orchestrator.state.session_id == "CALL-002"
    assert s1.orchestrator.state.verification.customer_name == "Priya"
    assert s2.orchestrator.state.verification.customer_name == "Ramesh"

    # Turn on session 1 does not affect session 2
    s1.orchestrator.process_turn("Yes this is Priya")
    assert s1.orchestrator.turn_count == 1
    assert s2.orchestrator.turn_count == 0


def test_session_idempotency_deduplication():
    """Duplicate event IDs must be detected and not process repeated turns."""
    sm = SessionManager(store=InMemorySessionStore())
    stub = StubLLMAdapter(replies=[
        'Hi Priya! ```json {"is_identity_confirmed": true, "is_available": true} ```'
    ])
    session = sm.get_or_create_session(call_id="CALL-IDEMPOTENT", llm=stub)

    # First turn with event_id
    res1 = sm.process_turn(
        call_id="CALL-IDEMPOTENT",
        customer_utterance="Yes I am Priya",
        event_id="EVT-100",
    )
    assert res1.state.stage == Stage.QUALIFICATION

    # Duplicate turn with same event_id
    res2 = sm.process_turn(
        call_id="CALL-IDEMPOTENT",
        customer_utterance="Yes I am Priya",
        event_id="EVT-100",
    )
    # Turn count should remain 1 on orchestrator
    assert session.orchestrator.turn_count == 1
    assert "listening" in res2.agent_reply.lower()


def test_session_ttl_cleanup():
    """Stale sessions older than TTL are automatically purged."""
    store = InMemorySessionStore()
    sm = SessionManager(store=store, ttl_seconds=1)

    s1 = sm.get_or_create_session(call_id="CALL-EXPIRE")
    assert store.count() == 1

    # Wait for TTL expiration
    time.sleep(1.1)

    # Trigger cleanup via new session lookup
    sm.get_or_create_session(call_id="CALL-NEW")
    assert store.get("CALL-EXPIRE") is None


def test_terminal_stage_protection():
    """Once call reaches a terminal stage, further turns return end disposition."""
    sm = SessionManager(store=InMemorySessionStore())
    stub = StubLLMAdapter(replies=[
        'Not available. ```json {"is_identity_confirmed": true, "is_available": false} ```'
    ])
    res1 = sm.process_turn(
        call_id="CALL-TERMINAL",
        customer_utterance="No I am busy right now",
        llm=stub,
    )
    assert res1.state.stage == Stage.CALLBACK_REQUESTED
    assert res1.action == "callback"

    # Subsequent turn after callback requested
    res2 = sm.process_turn(
        call_id="CALL-TERMINAL",
        customer_utterance="Hello are you there?",
    )
    assert res2.action == "end"
    assert "goodbye" in res2.agent_reply.lower()
