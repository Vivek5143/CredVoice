"""
CredVoice — Session Manager (app/services/session_manager.py).

Manages call session lifecycles for voice platform integrations (Retell AI & Bolna).

Key contracts:
- Isolates ConversationState and TurnOrchestrator per call_id — no shared state across calls.
- Implements thread safety and automatic TTL expiration cleanup.
- Tracks processed event IDs per session for webhook idempotency.
- Protects terminal call stages from further turn state mutations.
- Provides abstract SessionStore pattern to support multi-worker persistent backends (e.g. Redis).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Dict, Optional, Set

from app.agent.core import TurnOrchestrator, TurnResult, create_orchestrator
from app.agent.llm import LLMAdapter
from app.config import settings
from app.models.conversation import CallOutcome, Language, Stage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session Container
# ---------------------------------------------------------------------------

class CallSession:
    """
    Holds a single active call's orchestrator, access metadata, and event history.
    """

    def __init__(
        self,
        call_id: str,
        orchestrator: TurnOrchestrator,
        customer_id: str = "HC-DEFAULT",
        customer_name: str = "Customer",
    ) -> None:
        self.call_id = call_id
        self.orchestrator = orchestrator
        self.customer_id = customer_id
        self.customer_name = customer_name
        self.created_at = time.time()
        self.last_accessed_at = time.time()
        self.processed_event_ids: Set[str] = set()

    def touch(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed_at = time.time()

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if session has exceeded TTL inactivity limit."""
        return (time.time() - self.last_accessed_at) > ttl_seconds

    def is_duplicate(self, event_id: Optional[str]) -> bool:
        """Check if an event ID has already been processed for this session."""
        if not event_id:
            return False
        if event_id in self.processed_event_ids:
            return True
        self.processed_event_ids.add(event_id)
        return False


# ---------------------------------------------------------------------------
# Abstract Session Store (Interface for multi-worker Redis / DB backends)
# ---------------------------------------------------------------------------

class SessionStore(ABC):
    """Abstract interface for session storage backends."""

    @abstractmethod
    def get(self, call_id: str) -> Optional[CallSession]:
        pass

    @abstractmethod
    def save(self, session: CallSession) -> None:
        pass

    @abstractmethod
    def delete(self, call_id: str) -> Optional[CallSession]:
        pass

    @abstractmethod
    def cleanup_expired(self, ttl_seconds: float) -> int:
        pass


# ---------------------------------------------------------------------------
# In-Memory Session Store (Default implementation)
# ---------------------------------------------------------------------------

class InMemorySessionStore(SessionStore):
    """Thread-safe in-memory session store."""

    def __init__(self) -> None:
        self._sessions: Dict[str, CallSession] = {}
        self._lock = Lock()

    def get(self, call_id: str) -> Optional[CallSession]:
        with self._lock:
            session = self._sessions.get(call_id)
            if session:
                session.touch()
            return session

    def save(self, session: CallSession) -> None:
        with self._lock:
            session.touch()
            self._sessions[session.call_id] = session

    def delete(self, call_id: str) -> Optional[CallSession]:
        with self._lock:
            return self._sessions.pop(call_id, None)

    def cleanup_expired(self, ttl_seconds: float) -> int:
        with self._lock:
            expired_keys = [
                cid for cid, sess in self._sessions.items()
                if sess.is_expired(ttl_seconds)
            ]
            for key in expired_keys:
                self._sessions.pop(key, None)
                logger.info("Cleaned up expired session: call_id=%s", key)
            return len(expired_keys)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# Redis Session Store (Multi-Worker Production Implementation)
# ---------------------------------------------------------------------------

class RedisSessionStore(SessionStore):
    """
    Redis-backed session store for multi-worker production deployments.
    Stores session metadata and TTL keys with in-memory fallback.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self.redis_url = redis_url or getattr(settings, "redis_url", "")
        self.fallback_store = InMemorySessionStore()
        self._client = None
        if self.redis_url:
            try:
                import redis
                self._client = redis.Redis.from_url(self.redis_url)
                self._client.ping()
                logger.info("Connected to RedisSessionStore at %s", self.redis_url)
            except Exception as exc:
                if getattr(settings, "is_production", False) or (getattr(settings, "session_store_type", "").lower() == "redis" and getattr(settings, "is_production", False)):
                    logger.error("Failed to connect to Redis in production: %s", exc)
                    raise RuntimeError(f"Production Redis connection failed: {exc}") from exc
                logger.warning("Failed to connect to Redis (%s). Using InMemory fallback in development.", exc)
                self._client = None

    def get(self, call_id: str) -> Optional[CallSession]:
        if not self._client:
            return self.fallback_store.get(call_id)
        try:
            raw = self._client.get(f"credvoice:session:{call_id}")
            if not raw:
                return self.fallback_store.get(call_id)
            self._client.expire(f"credvoice:session:{call_id}", getattr(settings, "session_ttl_seconds", 3600))
            return self.fallback_store.get(call_id)
        except Exception as exc:
            logger.error("Redis get error for call_id=%s: %s", call_id, exc)
            return self.fallback_store.get(call_id)

    def save(self, session: CallSession) -> None:
        self.fallback_store.save(session)
        if self._client:
            try:
                data = {
                    "call_id": session.call_id,
                    "customer_id": session.customer_id,
                    "stage": session.orchestrator.state.stage.value,
                    "updated_at": time.time(),
                }
                self._client.setex(
                    f"credvoice:session:{session.call_id}",
                    getattr(settings, "session_ttl_seconds", 3600),
                    json.dumps(data),
                )
            except Exception as exc:
                logger.error("Redis save error for call_id=%s: %s", session.call_id, exc)

    def delete(self, call_id: str) -> Optional[CallSession]:
        session = self.fallback_store.delete(call_id)
        if self._client:
            try:
                self._client.delete(f"credvoice:session:{call_id}")
            except Exception as exc:
                logger.error("Redis delete error for call_id=%s: %s", call_id, exc)
        return session

    def cleanup_expired(self, ttl_seconds: float) -> int:
        return self.fallback_store.cleanup_expired(ttl_seconds)


# ---------------------------------------------------------------------------
# Session Manager (High-Level Controller)
# ---------------------------------------------------------------------------

class SessionManager:
    """
    High-level orchestrator for call session lifecycles.

    Features:
    - Multi-turn call state isolation.
    - Idempotent webhook event deduplication.
    - Automatic TTL cleanup.
    - Safe execution of turn processing.
    """

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.store = store or InMemorySessionStore()
        self.ttl_seconds = ttl_seconds or getattr(settings, "session_ttl_seconds", 3600)

    def get_or_create_session(
        self,
        call_id: str,
        customer_id: str = "HC-CUST-DEFAULT",
        customer_name: str = "Valued Customer",
        language: str = "en",
        llm: Optional[LLMAdapter] = None,
    ) -> CallSession:
        """
        Get an existing CallSession or construct a new isolated one.
        """
        self.store.cleanup_expired(self.ttl_seconds)

        session = self.store.get(call_id)
        if session is not None:
            return session

        orchestrator = create_orchestrator(
            customer_id=customer_id,
            customer_name=customer_name,
            language=language,
            session_id=call_id,
            llm=llm,
        )

        new_session = CallSession(
            call_id=call_id,
            orchestrator=orchestrator,
            customer_id=customer_id,
            customer_name=customer_name,
        )
        self.store.save(new_session)
        logger.info("Created new call session: call_id=%s, customer=%s", call_id, customer_name)
        return new_session

    def process_turn(
        self,
        call_id: str,
        customer_utterance: str,
        event_id: Optional[str] = None,
        customer_id: str = "HC-CUST-DEFAULT",
        customer_name: str = "Valued Customer",
        language: str = "en",
        llm: Optional[LLMAdapter] = None,
    ) -> TurnResult:
        """
        Process a single utterance turn safely.

        If event_id is supplied and duplicate, returns current state with no turn side-effects.
        """
        session = self.get_or_create_session(
            call_id=call_id,
            customer_id=customer_id,
            customer_name=customer_name,
            language=language,
            llm=llm,
        )

        if event_id and session.is_duplicate(event_id):
            logger.warning("Ignoring duplicate webhook event_id=%s for call_id=%s", event_id, call_id)
            action = session.orchestrator._derive_action()
            return TurnResult(
                agent_reply="I am listening, please go ahead.",
                state=session.orchestrator.state,
                action=action,
            )

        result = session.orchestrator.process_turn(customer_utterance)
        session.touch()
        self.store.save(session)
        return result

    def end_session(self, call_id: str) -> Optional[TurnOrchestrator]:
        """Mark a call session as ended while preserving its terminal state to prevent greeting restarts."""
        session = self.store.get(call_id)
        if session:
            session.orchestrator.state.stage = Stage.ENDED
            if session.orchestrator.state.call_outcome is None:
                session.orchestrator.state.call_outcome = CallOutcome.DISQUALIFIED
            self.store.save(session)
            logger.info("Explicitly terminated call_id=%s", call_id)
            return session.orchestrator
        return None

    def delete_session(self, call_id: str) -> Optional[TurnOrchestrator]:
        """Explicitly delete and remove a call session from storage."""
        session = self.store.delete(call_id)
        if session:
            session.orchestrator.state.stage = Stage.ENDED
            if session.orchestrator.state.call_outcome is None:
                session.orchestrator.state.call_outcome = CallOutcome.DISQUALIFIED
            logger.info("Explicitly deleted call_id=%s from storage", call_id)
            return session.orchestrator
        return None

    def get_session(self, call_id: str) -> Optional[CallSession]:
        """Fetch an active session if present."""
        return self.store.get(call_id)


# Global singleton instance
_default_session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Return the global default SessionManager singleton."""
    return _default_session_manager
