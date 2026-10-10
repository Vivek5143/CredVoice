"""
CredVoice — Prompt Builder.

Builds context-aware system prompts and parses structured customer responses.

Design principles:
- System prompt is constructed fresh each turn, injecting current state so the LLM
  always knows what has been collected and what remains outstanding.
- Structured extraction is the LLM's only job: it returns a JSON object mapping field
  names to extracted values. The eligibility engine, not the LLM, decides outcomes.
- Prompts are language-aware (English / Hindi).
- No business-rule logic lives here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Optional

from app.config import settings
from app.models.conversation import (
    CallbackPreference,
    ConversationState,
    EligibilityStatus,
    ExistingLoanTransfer,
    IncomeMode,
    Language,
    OccupationType,
    OwnershipType,
    PropertyType,
    QualificationState,
    Stage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_FIELD_LABELS_EN: dict[str, str] = {
    "property_type": "property type (residential / commercial / industrial / agricultural)",
    "ownership": "property ownership (sole / joint)",
    "has_original_documents": "original property documents availability (yes / no)",
    "requested_loan_amount_inr": "desired loan amount in rupees",
    "proceeds_at_cap": "consent to proceed with ₹75 lakh (yes / no)",
    "occupation_type": "occupation (salaried / self-employed)",
    "income_mode": "how income is received (bank transfer or cash in hand)",
    "property_market_value_inr": "estimated market value of the property in rupees",
    "requested_tenure_years": "preferred loan tenure in years (3 to 15)",
}

_FIELD_LABELS_HI: dict[str, str] = {
    "property_type": "संपत्ति का प्रकार (आवासीय / व्यावसायिक / औद्योगिक / कृषि)",
    "ownership": "संपत्ति का स्वामित्व (एकल / संयुक्त)",
    "has_original_documents": "मूल संपत्ति दस्तावेज़ उपलब्ध हैं (हाँ / नहीं)",
    "requested_loan_amount_inr": "इच्छित ऋण राशि रुपये में",
    "proceeds_at_cap": "₹75 लाख पर आगे बढ़ने की सहमति (हाँ / नहीं)",
    "occupation_type": "व्यवसाय (वेतनभोगी / स्व-रोज़गार)",
    "income_mode": "आय प्राप्त करने का तरीका (बैंक हस्तांतरण या नकद)",
    "property_market_value_inr": "संपत्ति का अनुमानित बाज़ार मूल्य रुपये में",
    "requested_tenure_years": "पसंदीदा ऋण अवधि वर्षों में (3 से 15)",
}

# ---------------------------------------------------------------------------
# Extraction schema — what the LLM must return as JSON
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA = {
    "type": "object",
    "description": (
        "Extract ONLY values explicitly stated by the customer in this turn. "
        "Omit any key whose value cannot be determined from the customer's words. "
        "Do not infer, assume, or hallucinate values. "
        "Do not override the eligibility decision — that is done by the rules engine."
    ),
    "properties": {
        # -- Routing signals (checked before qualification) --
        "callback_requested": {
            "type": "boolean",
            "description": "True if customer says they are busy and wants a call back.",
        },
        "callback_preferred_time": {
            "type": "string",
            "description": "Date/time string for preferred callback, e.g. 'tomorrow 3pm'.",
        },
        "callback_contact_number": {
            "type": "string",
            "description": "Phone number for callback if provided.",
        },
        "language_switch": {
            "type": "string",
            "enum": ["en", "hi"],
            "description": "Set if customer explicitly requests a language change.",
        },
        "has_existing_property_loan": {
            "type": "boolean",
            "description": "True if customer mentions they already have a property loan.",
        },
        "wants_emi_reduction": {
            "type": "boolean",
            "description": "True if customer wants to reduce EMI / do a balance transfer.",
        },
        # -- Identity / availability --
        "is_identity_confirmed": {
            "type": "boolean",
            "description": "True if customer confirms they are the named person.",
        },
        "is_available": {
            "type": "boolean",
            "description": "True if customer says it is a good time to speak.",
        },
        # -- Seven qualification fields --
        "property_type": {
            "type": "string",
            "enum": ["residential", "commercial", "industrial", "agricultural", "other"],
            "description": "Type of the property.",
        },
        "ownership": {
            "type": "string",
            "enum": ["sole", "joint"],
            "description": "Ownership structure of the property.",
        },
        "has_original_documents": {
            "type": "boolean",
            "description": "True if original property documents are available.",
        },
        "requested_loan_amount_inr": {
            "type": "number",
            "description": "Loan amount in INR as a number (e.g. 5000000 for ₹50 lakh).",
        },
        "proceeds_at_cap": {
            "type": "boolean",
            "description": "True if customer agrees to proceed with ₹75 lakh when amount exceeded the cap.",
        },
        "occupation_type": {
            "type": "string",
            "enum": ["salaried", "self_employed", "unemployed", "other"],
        },
        "income_mode": {
            "type": "string",
            "enum": ["bank", "cash"],
            "description": "bank = bank transfer / cheque / online; cash = cash in hand only.",
        },
        "property_market_value_inr": {
            "type": "number",
            "description": "Customer's estimate of property market value in INR.",
        },
        "requested_tenure_years": {
            "type": "integer",
            "description": "Preferred tenure in whole years.",
        },
        "faq_answered": {
            "type": "boolean",
            "description": "Set to true if this turn was primarily answering a product/FAQ question.",
        },
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def _load_base_prompt(language: Language) -> str:
    """Load base system prompt from prompts/ directory, fall back to inline text."""
    fname = "system_en.txt" if language == Language.ENGLISH else "system_hi.txt"
    fpath = _PROMPTS_DIR / fname
    if fpath.exists():
        return fpath.read_text(encoding="utf-8").strip()
    # Inline fallback — used if prompt files are missing
    return (
        "You are CredVoice, a professional voice agent for Home Credit India. "
        "You are calling an existing customer about a pre-approved Loan Against Property "
        "offer of up to ₹75 lakh. Be polite, clear, and concise."
    )


def build_system_prompt(
    state: ConversationState,
    faq_context: Optional[str] = None,
) -> str:
    """
    Construct the full system prompt for the current conversation turn.

    The prompt is re-built on every turn so the LLM always has accurate state.
    It instructs the LLM on:
    - Its role and language
    - What has already been collected (do NOT ask again)
    - What is still missing (ask naturally — do not enumerate all at once)
    - Special handling for cap consent, disqualification, transfer routing, callbacks
    - Grounded FAQ context for the current customer question (if a match was found)
    - Extraction schema description (the LLM must return a JSON extraction object)

    Args:
        state: Current conversation state.
        faq_context: Optional pre-fetched FAQ answer to inject for this turn.
                     When provided the LLM is instructed to use it verbatim.

    Returns:
        Complete system prompt string.
    """
    lang = state.language
    labels = _FIELD_LABELS_EN if lang == Language.ENGLISH else _FIELD_LABELS_HI
    base = _load_base_prompt(lang)

    # ---- Collected fields summary ----
    q = state.qualification
    collected: list[str] = []
    if q.property_type is not None:
        collected.append(f"property_type = {q.property_type.value}")
    if q.ownership is not None:
        collected.append(f"ownership = {q.ownership.value}")
    if q.has_original_documents is not None:
        collected.append(f"has_original_documents = {q.has_original_documents}")
    if q.requested_loan_amount_inr is not None:
        collected.append(f"requested_loan_amount_inr = {q.requested_loan_amount_inr:,.0f}")
    if q.proceeds_at_cap is not None:
        collected.append(f"proceeds_at_cap = {q.proceeds_at_cap}")
    if q.occupation_type is not None:
        collected.append(f"occupation_type = {q.occupation_type.value}")
    if q.income_mode is not None:
        collected.append(f"income_mode = {q.income_mode.value}")
    if q.property_market_value_inr is not None:
        collected.append(f"property_market_value_inr = {q.property_market_value_inr:,.0f}")
    if q.requested_tenure_years is not None:
        collected.append(f"requested_tenure_years = {q.requested_tenure_years}")

    collected_block = (
        "\n".join(f"  • {c}" for c in collected) if collected else "  (none yet)"
    )

    # ---- Missing fields with human labels ----
    missing_labels = [labels.get(f, f) for f in state.missing_fields]
    missing_block = (
        "\n".join(f"  • {m}" for m in missing_labels) if missing_labels else "  (none)"
    )

    # ---- Stage-specific instructions ----
    stage_instructions = _stage_instructions(state, lang)

    # ---- FAQ context block ----
    faq_block = ""
    if faq_context:
        faq_block = (
            "\n--- GROUNDED PRODUCT CONTEXT FOR THIS TURN ---\n"
            f"{faq_context}\n"
            "Use the above answer to respond to the customer's product question. "
            "Do not add information not present in the answer above. "
            "After answering, return naturally to collecting the next outstanding qualification field."
        )

    # ---- Extraction instruction ----
    extraction_instruction = (
        "\n\nAt the end of every reply you MUST include a JSON block surrounded by "
        "```json and ``` markers containing ONLY the fields the customer explicitly "
        "stated in THIS turn. Use the extraction schema below. "
        "Omit keys that were not mentioned. Never fabricate values.\n"
        f"Schema:\n{json.dumps(EXTRACTION_SCHEMA, ensure_ascii=False, indent=2)}"
    )

    prompt = f"""{base}

---

CURRENT CONVERSATION STATE
Stage        : {state.stage.value}
Language     : {lang.value}
Eligibility  : {state.eligibility_status.value}

Already collected (DO NOT ask for these again):
{collected_block}

Still needed:
{missing_block}

--- STAGE-SPECIFIC INSTRUCTIONS ---
{stage_instructions}

--- GENERAL RULES ---
• Never ask for a field already collected.
• Accept answers given out of order or volunteered mid-sentence.
• If the customer corrects a previous answer, accept the correction.
• Never make eligibility promises — the rules engine decides.
• Never invent Home Credit policies, interest rates, or eligibility thresholds.
• Ask for only ONE outstanding field per turn unless the customer volunteers several.
• If the customer is ambiguous, ask a single clarifying question.
{faq_block}
{extraction_instruction}"""
    return prompt


def _stage_instructions(state: ConversationState, lang: Language) -> str:
    """Return stage-specific behavioural instructions embedded in the system prompt."""
    stage = state.stage
    q = state.qualification

    if stage == Stage.VERIFICATION:
        return (
            "Greet the customer, confirm their name, and ask whether now is a good time to talk. "
            "If they confirm identity and availability, transition to QUALIFICATION. "
            "If they are busy, offer a callback and capture their preferred time. "
            "If the person is not the named customer, politely end the call."
        )

    if stage == Stage.QUALIFICATION:
        cap_note = ""
        if q.requires_amount_cap_consent and q.proceeds_at_cap is None:
            cap_note = (
                "\n⚠ CAP CONSENT REQUIRED: The customer requested more than ₹75 lakh. "
                "Explain the cap clearly and ask whether they wish to proceed with ₹75 lakh. "
                "Do NOT continue collecting other fields until consent is obtained."
            )
        return (
            "Collect the remaining eligibility information naturally. "
            "Do NOT ask about fields already collected. "
            "Handle out-of-order answers gracefully. "
            "If the customer corrects an earlier answer, note the correction in the JSON extraction."
            + cap_note
        )

    if stage == Stage.DISQUALIFIED:
        reasons = ", ".join(r.value for r in state.disqualification_reasons)
        return (
            f"The customer has been disqualified ({reasons}). "
            "Inform them politely and concisely. Do NOT ask any further qualification questions. "
            "Thank them for their time and end the call."
        )

    if stage == Stage.TRANSFER_ROUTED:
        return (
            "The customer has an existing property loan or wants EMI reduction. "
            "Inform them a Loan-Transfer Specialist will assist them and provide a warm handoff. "
            "Do NOT continue with new-loan qualification questions."
        )

    if stage == Stage.ELIGIBLE_HANDOFF:
        return (
            "All qualification criteria are met. "
            "Congratulate the customer warmly and inform them a Senior Loan Expert will now assist them. "
            "Provide a warm handoff. Do NOT ask further qualification questions."
        )

    if stage == Stage.CALLBACK_REQUESTED:
        return (
            "The customer has requested a callback. "
            "Confirm the preferred time and contact number. "
            "Thank them and end the call gracefully."
        )

    return "Continue the conversation per general rules above."


# ---------------------------------------------------------------------------
# Response extraction (parse LLM output)
# ---------------------------------------------------------------------------

def extract_json_from_response(llm_output: str) -> dict:
    """
    Parse the JSON extraction block from the LLM's reply.

    Searches for a ```json ... ``` fenced block. Falls back to raw JSON parse
    if no fences are found. Returns an empty dict on failure.

    Args:
        llm_output: Raw LLM response text.

    Returns:
        Parsed extraction dict or empty dict if parsing fails.
    """
    if not llm_output or not isinstance(llm_output, str):
        return {}

    # Try fenced block first
    start_marker = "```json"
    end_marker = "```"
    start_idx = llm_output.find(start_marker)
    if start_idx != -1:
        start_idx += len(start_marker)
        end_idx = llm_output.find(end_marker, start_idx)
        if end_idx != -1:
            raw = llm_output[start_idx:end_idx].strip()
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
                logger.warning("Fenced JSON block content is not a JSON object: %s", type(data))
                return {}
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse fenced JSON block: %s", exc)
                return {}

    # Fallback: attempt to parse entire output as JSON
    try:
        data = json.loads(llm_output.strip())
        if isinstance(data, dict):
            return data
        logger.warning("Parsed JSON content is not a JSON object: %s", type(data))
        return {}
    except json.JSONDecodeError:
        logger.warning("LLM output contained no parseable JSON extraction block.")
        return {}


def normalize_utterance_extraction(
    utterance: str,
    existing_extraction: Optional[dict] = None,
) -> dict:
    """
    Deterministically normalize customer utterances (Hindi & English) into structured fields.
    Normalizes phonetic and ASR variations (e.g. 'क   ष    भ   म    ', 'ख   त    क    जम   न ',
    'नकद नकद') to standard enum values.
    """
    res = dict(existing_extraction or {})
    if not utterance or not isinstance(utterance, str):
        return res

    text = utterance.strip()
    text_lower = text.lower()
    collapsed = re.sub(r'[\s\-_,.]+', '', text_lower)

    # 1. Language auto-detection (Devanagari script presence)
    if re.search(r'[\u0900-\u097F]', text) and "language_switch" not in res:
        res["language_switch"] = "hi"

    # 2. Verification / Identity & Availability Normalization
    if any(p in collapsed for p in ["हाँ", "हां", "बोलरहाहूँ", "बोलरहाहु", "बातकरसकतेहैं", "सहीसमयहै", "मैंहीहूँ"]) or any(
        w in text_lower for w in ["yes", "speaking", "good time to speak", "can talk", "go ahead"]
    ):
        if "is_identity_confirmed" not in res:
            res["is_identity_confirmed"] = True
        if "is_available" not in res:
            res["is_available"] = True

    # 3. Property Type Normalization
    agri_patterns = ["कषभम", "कृषि", "खेत", "खेती", "खतकजमन"]
    if any(p in collapsed for p in agri_patterns) or any(
        w in text_lower for w in ["agricultural", "farmland", "agriculture", "farm land"]
    ):
        res["property_type"] = "agricultural"
    elif "property_type" not in res:
        is_office_inquiry = any(term in text_lower for term in ["head office", "office address", "post office", "branch office"])
        if any(p in collapsed for p in ["आवासीय", "मकान"]) or re.search(r'\b(?:residential|house|flat|apartment)\b', text_lower):
            res["property_type"] = "residential"
        elif not is_office_inquiry and (
            any(p in collapsed for p in ["व्यावसायिक", "दुकान", "शोरूम", "कमर्शियल"])
            or re.search(r'\b(?:commercial|shop|showroom)\b', text_lower)
            or any(term in text_lower for term in ["office space", "office building", "office property"])
        ):
            res["property_type"] = "commercial"
        elif any(p in collapsed for p in ["औद्योगिक", "कारखाना"]) or re.search(r'\b(?:industrial|factory|warehouse)\b', text_lower):
            res["property_type"] = "industrial"

    # 3. Income Mode Normalization
    cash_patterns = ["नकद", "नगद", "रोकड़", "रोकड", "रोखड़"]
    if any(p in collapsed for p in cash_patterns) or re.search(r'\bcash\b', text_lower):
        res["income_mode"] = "cash"
    elif "income_mode" not in res:
        bank_patterns = ["बैंक", "खातेमें", "खाता"]
        if any(p in collapsed for p in bank_patterns) or any(w in text_lower for w in ["bank", "account", "online", "cheque"]):
            res["income_mode"] = "bank"

    # 4. Ownership Normalization
    is_ownership_question = bool(re.search(r'\b(?:is|can|are)\b.*\b(?:allowed|permitted|eligible|possible)\b|\?', text_lower))
    if "ownership" not in res and not is_ownership_question:
        if any(p in collapsed for p in ["एकल", "खुदका", "मेरेनाम", "मेराही"]) or re.search(r'\b(?:sole|single|only mine)\b', text_lower):
            res["ownership"] = "sole"
        elif any(p in collapsed for p in ["संयुक्त", "पत्नीकेसाथ", "भाईकेसाथ"]) or re.search(r'\b(?:joint|jointly)\b', text_lower):
            res["ownership"] = "joint"

    # 5. Original Documents Availability
    if "has_original_documents" not in res:
        if any(p in collapsed for p in ["मूलदस्तावेज़", "ओरिजिनलदस्तावेज़", "मूलकागज़ात", "कागज़ातहैं", "कागजातहैं"]):
            res["has_original_documents"] = True
        elif any(w in text_lower for w in ["original documents available", "have original papers"]):
            res["has_original_documents"] = True

    # 6. Existing Loan / EMI Reduction Transfer Intent
    existing_loan_patterns = ["पुरानालोन", "चलरहालोन", "मौजूदालोन", "पहलेसेलोन"]
    if any(p in collapsed for p in existing_loan_patterns) or any(w in text_lower for w in ["existing loan", "already have a loan", "loan on property"]):
        res["has_existing_property_loan"] = True

    emi_reduction_patterns = ["emiकम", "ईएमआईकम", "किश्तकम"]
    if any(p in collapsed for p in emi_reduction_patterns) or any(w in text_lower for w in ["emi reduction", "reduce emi", "lower emi", "balance transfer"]):
        res["wants_emi_reduction"] = True

    # 7. ₹75 Lakh Cap Consent
    if "proceeds_at_cap" not in res:
        if any(p in collapsed for p in ["75लाखचलेगा", "75लाखठीकहै", "75लाखमंजूर", "सहमतिहै"]) or any(w in text_lower for w in ["75 lakh is fine", "accept 75 lakh", "proceed with 75 lakh"]):
            res["proceeds_at_cap"] = True
        elif any(p in collapsed for p in ["75लाखनहीं", "कमहै", "सहमतिनहीं"]) or any(w in text_lower for w in ["cannot proceed with 75 lakh", "need more than 75 lakh"]):
            res["proceeds_at_cap"] = False

    # 8. Tenure Normalization
    tenure_match = re.search(r'\b(\d{1,2})\s*(?:years?|yrs?|साल|वर्ष)\b', text_lower)
    if not tenure_match:
        standalone_match = re.fullmatch(r'(\d{1,2})', text_lower.strip())
        if standalone_match:
            tenure_match = standalone_match

    if tenure_match and "requested_tenure_years" not in res:
        try:
            res["requested_tenure_years"] = int(tenure_match.group(1))
        except (ValueError, TypeError):
            pass

    return res


def apply_extraction_to_state(
    extraction: dict,
    state: ConversationState,
    customer_utterance: Optional[str] = None,
) -> ConversationState:
    """
    Apply extracted field values onto the conversation state.
    Optionally normalizes customer_utterance deterministically first.

    Only fields present in the extraction dict are applied. Existing values are
    overwritten if the customer corrected them. The eligibility engine is NOT
    called here — that is the orchestrator's responsibility after this function.

    Args:
        extraction: Parsed extraction dict from extract_json_from_response().
        state: Mutable ConversationState to update.
        customer_utterance: Optional raw utterance string for deterministic normalization.

    Returns:
        Updated ConversationState (same object, mutated in place).
    """
    if customer_utterance:
        extraction = normalize_utterance_extraction(customer_utterance, extraction)

    # -- Routing / meta fields --
    if extraction.get("callback_requested") is True:
        state.callback.callback_requested = True
        state.stage = Stage.CALLBACK_REQUESTED

    if "callback_preferred_time" in extraction:
        state.callback.preferred_time = extraction["callback_preferred_time"]

    if "callback_contact_number" in extraction:
        state.callback.contact_number = extraction["callback_contact_number"]

    if "language_switch" in extraction:
        raw_lang = extraction["language_switch"]
        try:
            state.language = Language(raw_lang)
        except ValueError:
            logger.warning("Unknown language switch value: %s", raw_lang)

    # -- Identity / availability --
    if "is_identity_confirmed" in extraction:
        state.verification.is_identity_verified = extraction["is_identity_confirmed"]

    if "is_available" in extraction:
        state.verification.is_available_to_speak = extraction["is_available"]
        if extraction["is_available"] is False:
            state.callback.callback_requested = True

    # -- Existing loan / EMI transfer --
    if "has_existing_property_loan" in extraction:
        state.transfer_intent.has_existing_property_loan = extraction["has_existing_property_loan"]

    if "wants_emi_reduction" in extraction:
        state.transfer_intent.wants_emi_reduction = extraction["wants_emi_reduction"]

    # -- Seven qualification fields --
    q = state.qualification

    if "property_type" in extraction:
        raw = extraction["property_type"]
        try:
            q.property_type = PropertyType(raw)
        except ValueError:
            logger.warning("Unknown property_type value: %s", raw)

    if "ownership" in extraction:
        raw = extraction["ownership"]
        try:
            q.ownership = OwnershipType(raw)
        except ValueError:
            logger.warning("Unknown ownership value: %s", raw)

    if "has_original_documents" in extraction:
        q.has_original_documents = bool(extraction["has_original_documents"])

    if "requested_loan_amount_inr" in extraction:
        try:
            q.requested_loan_amount_inr = float(extraction["requested_loan_amount_inr"])
        except (TypeError, ValueError):
            logger.warning("Invalid requested_loan_amount_inr: %s", extraction["requested_loan_amount_inr"])

    if "proceeds_at_cap" in extraction:
        q.proceeds_at_cap = bool(extraction["proceeds_at_cap"])

    if "occupation_type" in extraction:
        raw = extraction["occupation_type"]
        try:
            q.occupation_type = OccupationType(raw)
        except ValueError:
            logger.warning("Unknown occupation_type value: %s", raw)

    if "income_mode" in extraction:
        raw = extraction["income_mode"]
        try:
            q.income_mode = IncomeMode(raw)
        except ValueError:
            logger.warning("Unknown income_mode value: %s", raw)

    if "property_market_value_inr" in extraction:
        try:
            q.property_market_value_inr = float(extraction["property_market_value_inr"])
        except (TypeError, ValueError):
            logger.warning("Invalid property_market_value_inr: %s", extraction["property_market_value_inr"])

    if "requested_tenure_years" in extraction:
        try:
            q.requested_tenure_years = int(extraction["requested_tenure_years"])
        except (TypeError, ValueError):
            logger.warning("Invalid requested_tenure_years: %s", extraction["requested_tenure_years"])

    return state
