"""
CredVoice Centralized Application Configuration.

Loads environment settings with sensible defaults for local development.
Secrets are read from environment variables or .env file and never hardcoded.
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # General Application Settings
    app_env: str = "development"
    service_name: str = "CredVoice"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # LLM Configuration
    llm_provider: str = "openai"  # e.g., 'openai', 'gemini', 'stub'
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    gemini_api_key: str = ""  # GEMINI_API_KEY env var — used when LLM_PROVIDER=gemini

    # Voice Platform Integration Configuration
    voice_provider: str = "retell"  # 'retell' or 'bolna'
    retell_api_key: str = ""
    retell_webhook_secret: str = ""
    bolna_api_key: str = ""
    bolna_webhook_secret: str = ""
    public_webhook_base_url: str = "http://localhost:8000"
    session_ttl_seconds: int = 3600
    verify_webhook_signatures: bool = False
    admin_api_key: str = ""
    cors_origins: List[str] = ["*"]
    max_request_body_bytes: int = 1048576  # 1MB request limit
    redis_url: str = ""
    session_store_type: str = "memory"  # 'memory' or 'redis'

    @property
    def is_production(self) -> bool:
        """Check if application environment is production."""
        return self.app_env.lower() in ("production", "prod")

    @property
    def should_verify_signatures(self) -> bool:
        """Enforce webhook signature verification in production or when enabled."""
        return self.is_production or self.verify_webhook_signatures

    # Language Configuration
    default_language: str = "en"  # 'en' or 'hi'
    supported_languages: List[str] = ["en", "hi"]

    # Product Knowledge / RAG Settings
    enable_rag: bool = True
    knowledge_base_path: str = "knowledge/lap_faq.json"

    # Qualification Constraints (Home Credit LAP Offer Cap)
    max_loan_amount_lakh: float = 75.0
    max_loan_amount_inr: float = 7500000.0  # ₹75 Lakh (7,500,000 INR)
    min_tenure_years: int = 3
    max_tenure_years: int = 15


# Global singleton settings instance
settings = Settings()
