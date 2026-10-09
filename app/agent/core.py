"""
CredVoice — Turn Orchestrator (core.py).

Manages a single conversation session from call start to resolution:
verification → qualification → disqualification / transfer / handoff / callback.

Design contract:
- The orchestrator is the single owner of ConversationState mutation.
- Business rules (eligibility) are always evaluated by EligibilityEngine after
  every state update. The LLM output is never the source of truth for decisions.
- LLM calls are isolated behind process_customer_utterance() so they can be
  stubbed or replaced during testing without changing orchestrator logic.
- The orchestrator does NOT make HTTP calls to the voice platform — that is the
  integrations layer's responsibility.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from app.agent.prompt_builder import (
    apply_extraction_to_state,
    build_system_prompt,
    extract_json_from_response,
)
from app.agent.llm import (
    LLMAdapter,
    StubLLMAdapter,
    OpenAILLMAdapter,
    LLMError,
    LLMConfigurationError,
    create_llm_adapter,
)
from app.models.conversation import (
    CallOutcome,
    ConversationState,
    EligibilityStatus,
    Language,
    Stage,
)
from app.services.eligibility import EligibilityEngine, update_conversation_eligibility
from app.services.knowledge import KnowledgeService, KnowledgeResult, get_knowledge_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn result — what the orchestrator returns to the caller after each turn
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """
    Result of processing one customer utterance.

    Attributes:
        agent_reply:  The text response to be synthesised as speech.
        state:        Updated conversation state after this turn.
        action:       Disposition instruction for the caller
                      ("continue" | "handoff" | "transfer" | "callback" | "end").
        extraction:   Raw extraction dict parsed from the LLM output (for logging).
        faq_answered: True if this turn returned a grounded FAQ answer.
        faq_result:   The KnowledgeResult used for this turn (None if no FAQ lookup).
    """
    agent_reply: str
    state: ConversationState
    action: str
    extraction: dict = field(default_factory=dict)
    faq_answered: bool = False
    faq_result: object = None  # KnowledgeResult | None


# ---------------------------------------------------------------------------
# LLM adapter interface — swap implementations without touching the orchestrator
# ---------------------------------------------------------------------------

# Re-exported from app.agent.llm for backwards compatibility
__all_llm__ = ["LLMAdapter", "StubLLMAdapter", "OpenAILLMAdapter", "create_llm_adapter"]


# ---------------------------------------------------------------------------
# Turn orchestrator
# ---------------------------------------------------------------------------

class TurnOrchestrator:
    """
    Manages the full lifecycle of one CredVoice call session.

    Usage:
        orchestrator = TurnOrchestrator(
            customer_id="HC-12345",
            customer_name="Priya Sharma",
            language=Language.ENGLISH,
            llm=some_llm_adapter,
        )
        result = orchestrator.process_turn("Yes, this is a good time to speak.")
        print(result.agent_reply)
        print(result.action)
    """

    # Maximum turns allowed before forcing a graceful end (safety guard)
    MAX_TURNS: int = 40

    def __init__(
        self,
        customer_id: str,
        customer_name: str,
        language: Language = Language.ENGLISH,
        session_id: Optional[str] = None,
        llm: Optional[LLMAdapter] = None,
        knowledge_service: Optional[KnowledgeService] = None,
    ) -> None:
        self._session_id = session_id or str(uuid.uuid4())
        self._llm = llm
        self._knowledge = knowledge_service or get_knowledge_service()
        self._turn_count = 0

        # Initialise conversation state
        self.state = ConversationState(
            session_id=self._session_id,
            customer_id=customer_id,
            language=language,
            stage=Stage.VERIFICATION,
        )
        self.state.verification.customer_name = customer_name

        # Conversation history (OpenAI-style message list)
        self._history: list[dict[str, str]] = []

        logger.info(
            "Session %s started for customer %s (lang=%s)",
            self._session_id,
            customer_id,
            language.value,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_turn(self, customer_utterance: str) -> TurnResult:
        """
        Process a single customer utterance and return the agent's reply.

        Steps:
        1. Guard: if session is already terminated, reject further turns.
        2. Append utterance to history.
        3. Build system prompt from current state.
        4. Call LLM → raw reply text.
        5. Extract JSON from reply → apply to state.
        6. Run eligibility engine → update state stage.
        7. Derive action disposition.
        8. Append agent reply to history.
        9. Return TurnResult.

        Args:
            customer_utterance: Transcribed or typed customer input.

        Returns:
            TurnResult with agent_reply, updated state, and action disposition.
        """
        if self._is_terminal_stage():
            return TurnResult(
                agent_reply="Thank you for your time. Goodbye.",
                state=self.state,
                action="end",
            )

        self._turn_count += 1
        if self._turn_count > self.MAX_TURNS:
            logger.warning("Session %s exceeded MAX_TURNS — forcing end.", self._session_id)
            self.state.stage = Stage.ENDED
            return TurnResult(
                agent_reply="I'm sorry, let's connect again at a convenient time. Goodbye.",
                state=self.state,
                action="end",
            )

        self._history.append({"role": "user", "content": customer_utterance})

        # FAQ lookup — run before building prompt so context can be injected
        faq_result: KnowledgeResult = self._knowledge.lookup(
            utterance=customer_utterance,
            language=self.state.language.value,
        )
        faq_context = faq_result.answer if faq_result.found else None

        # Build prompt (injects FAQ context when available)
        system_prompt = build_system_prompt(self.state, faq_context=faq_context)

        # Call LLM
        raw_reply = self._call_llm(system_prompt, self._history)

        # Extract structured data from reply
        extraction = extract_json_from_response(raw_reply)
        logger.debug("Turn %d extraction: %s", self._turn_count, json.dumps(extraction))

        # Apply extraction to state (pure mutation, no eligibility check yet)
        apply_extraction_to_state(extraction, self.state)

        # After extraction, advance stage from VERIFICATION if appropriate
        self._advance_verification_stage()

        # Run eligibility engine (deterministic — overrides any LLM eligibility claim)
        if self.state.stage == Stage.QUALIFICATION:
            update_conversation_eligibility(self.state)

        # Determine action disposition
        action = self._derive_action()

        # Set call outcome for terminal stages
        self._set_call_outcome(action)

        # Strip the JSON block from the agent's visible reply
        visible_reply = self._strip_json_block(raw_reply)

        self._history.append({"role": "assistant", "content": visible_reply})

        # faq_answered is True when FAQ was looked up AND LLM signalled it in extraction
        faq_answered = faq_result.found or bool(extraction.get("faq_answered"))

        logger.info(
            "Session %s turn %d — stage=%s action=%s faq=%s",
            self._session_id,
            self._turn_count,
            self.state.stage.value,
            action,
            faq_answered,
        )

        return TurnResult(
            agent_reply=visible_reply,
            state=self.state,
            action=action,
            extraction=extraction,
            faq_answered=faq_answered,
            faq_result=faq_result,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, history: list[dict]) -> str:
        """Delegate to the injected LLM adapter or create default; handle failures gracefully."""
        if self._llm is None:
            try:
                self._llm = create_llm_adapter()
            except LLMError as exc:
                logger.warning("No valid LLM configuration available (%s). Returning fallback response.", exc)
                if self.state.language == Language.HINDI:
                    return (
                        "माफ़ कीजिए, तकनीकी समस्या के कारण उत्तर देने में समस्या आ रही है। "
                        "एक वरिष्ठ ऋण विशेषज्ञ जल्द ही आपसे संपर्क करेंगे। ```json {} ```"
                    )
                return (
                    "I am sorry, I am experiencing a temporary technical difficulty. "
                    "A Senior Loan Expert will assist you shortly. ```json {} ```"
                )

        try:
            return self._llm.chat(system_prompt=system_prompt, messages=list(history))
        except (LLMError, Exception) as exc:
            logger.error("LLM call failed in session %s: %s", self._session_id, exc)
            if self.state.language == Language.HINDI:
                return (
                    "माफ़ कीजिए, तकनीकी समस्या के कारण उत्तर देने में समस्या आ रही है। "
                    "एक वरिष्ठ ऋण विशेषज्ञ जल्द ही आपसे संपर्क करेंगे। ```json {} ```"
                )
            return (
                "I am sorry, I am experiencing a temporary technical difficulty. "
                "A Senior Loan Expert will assist you shortly. ```json {} ```"
            )

    def _is_terminal_stage(self) -> bool:
        return self.state.stage in {
            Stage.ENDED,
            Stage.DISQUALIFIED,
            Stage.ELIGIBLE_HANDOFF,
            Stage.TRANSFER_ROUTED,
            Stage.CALLBACK_REQUESTED,
        }

    def _advance_verification_stage(self) -> None:
        """
        Transition VERIFICATION → QUALIFICATION when both conditions are met:
        - Customer confirmed identity
        - Customer confirmed availability

        Also handles the case where the customer is unavailable (→ CALLBACK_REQUESTED).
        """
        if self.state.stage != Stage.VERIFICATION:
            return

        v = self.state.verification
        if v.is_identity_verified is False:
            # Wrong person — end the call
            self.state.stage = Stage.ENDED
            return

        if v.is_available_to_speak is False or self.state.callback.callback_requested:
            self.state.stage = Stage.CALLBACK_REQUESTED
            return

        if v.is_identity_verified is True and v.is_available_to_speak is True:
            self.state.stage = Stage.QUALIFICATION

    def _derive_action(self) -> str:
        """Map current stage to action string for the caller."""
        stage = self.state.stage
        if stage == Stage.ELIGIBLE_HANDOFF:
            return "handoff"
        if stage == Stage.TRANSFER_ROUTED:
            return "transfer"
        if stage == Stage.CALLBACK_REQUESTED:
            return "callback"
        if stage in {Stage.DISQUALIFIED, Stage.ENDED}:
            return "end"
        return "continue"

    def _set_call_outcome(self, action: str) -> None:
        """Record the final call outcome when a terminal state is reached."""
        if self.state.call_outcome is not None:
            return  # Already set
        outcome_map = {
            "handoff": CallOutcome.QUALIFIED_HANDOFF,
            "transfer": CallOutcome.TRANSFER_ROUTED,
            "callback": CallOutcome.CALLBACK_SCHEDULED,
            "end": CallOutcome.DISQUALIFIED,
        }
        if action in outcome_map:
            self.state.call_outcome = outcome_map[action]

    @staticmethod
    def _strip_json_block(text: str) -> str:
        """Remove the ```json ... ``` extraction block from the visible agent reply."""
        start = text.find("```json")
        if start == -1:
            return text.strip()
        end = text.find("```", start + 6)
        if end == -1:
            return text[:start].strip()
        before = text[:start].strip()
        after = text[end + 3:].strip()
        return (before + "\n" + after).strip() if after else before


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_orchestrator(
    customer_id: str,
    customer_name: str,
    language: str = "en",
    llm: Optional[LLMAdapter] = None,
    session_id: Optional[str] = None,
    knowledge_service: Optional[KnowledgeService] = None,
) -> TurnOrchestrator:
    """
    Factory function for creating a TurnOrchestrator from raw string inputs.

    Args:
        customer_id: Unique customer identifier.
        customer_name: Full name of the pre-approved customer.
        language: "en" or "hi".
        llm: LLMAdapter instance. Required for live calls; omit for tests.
        session_id: Optional session identifier (generated if omitted).
        knowledge_service: Optional KnowledgeService to inject (defaults to singleton).

    Returns:
        Configured TurnOrchestrator.
    """
    try:
        lang = Language(language)
    except ValueError:
        logger.warning("Unknown language '%s', defaulting to English.", language)
        lang = Language.ENGLISH

    return TurnOrchestrator(
        customer_id=customer_id,
        customer_name=customer_name,
        language=lang,
        session_id=session_id,
        llm=llm,
        knowledge_service=knowledge_service,
    )

