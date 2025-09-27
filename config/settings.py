"""
Configuration management for the multi-agent healthcare system.
Handles environment variables and system settings with multi-model provider support.
"""

import os
from typing import Optional, List, Dict
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelProviderConfig(BaseSettings):
    """Multi-model provider configuration for best accuracy."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra='ignore'  # Ignore extra fields from environment
    )

    # All providers use Groq API with different models for optimal performance
    # OpenAI Configuration (Using working Groq model)
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    openai_model: str = Field("openai/gpt-oss-20b", alias="OPENAI_MODEL")

    # Google Configuration (Using working Groq model)
    google_api_key: Optional[str] = Field(None, alias="GOOGLE_API_KEY")
    google_model: str = Field("openai/gpt-oss-20b", alias="GOOGLE_MODEL")

    # Groq Configuration (Primary - working model)
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("openai/gpt-oss-20b", alias="GROQ_MODEL")

    # DeepSeek Configuration (Using working Groq model)
    deepseek_api_key: Optional[str] = Field(None, alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field("openai/gpt-oss-20b", alias="DEEPSEEK_MODEL")

    # Alibaba Cloud Configuration (Using working Groq model)
    alibaba_api_key: Optional[str] = Field(None, alias="ALIBABA_API_KEY")
    alibaba_model: str = Field("openai/gpt-oss-20b", alias="ALIBABA_MODEL")

    # Meta Configuration (Using working Groq model)
    meta_api_key: Optional[str] = Field(None, alias="META_API_KEY")
    meta_model: str = Field("openai/gpt-oss-20b", alias="META_MODEL")

    # Moonshot AI Configuration (Using working Groq model)
    moonshot_api_key: Optional[str] = Field(None, alias="MOONSHOT_API_KEY")
    moonshot_model: str = Field("openai/gpt-oss-20b", alias="MOONSHOT_MODEL")

    # PlayAI Configuration (Using working Groq model)
    playai_api_key: Optional[str] = Field(None, alias="PLAYAI_API_KEY")
    playai_model: str = Field("openai/gpt-oss-20b", alias="PLAYAI_MODEL")

    # Model Selection Strategy
    primary_provider: str = Field("openai", alias="PRIMARY_MODEL_PROVIDER")  # Highest accuracy
    fallback_provider: str = Field("groq", alias="FALLBACK_MODEL_PROVIDER")  # Speed
    vision_provider: str = Field("google", alias="VISION_MODEL_PROVIDER")    # Complex analysis
    voice_provider: str = Field("openai", alias="VOICE_MODEL_PROVIDER")      # Voice processing
    diagnostic_provider: str = Field("openai", alias="DIAGNOSTIC_MODEL_PROVIDER")  # Medical diagnosis

    def get_provider_config(self, provider_name: str) -> Dict:
        """Get configuration for a specific provider (all using Groq API)."""

        # All providers use Groq's base URL since all API keys are Groq keys
        base_config = {
            "base_url": "https://api.groq.com/openai/v1",
            "headers": {"Content-Type": "application/json"}
        }

        provider_configs = {
            "openai": {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                **base_config
            },
            "google": {
                "api_key": self.google_api_key,
                "model": self.google_model,
                **base_config
            },
            "groq": {
                "api_key": self.groq_api_key,
                "model": self.groq_model,
                **base_config
            },
            "deepseek": {
                "api_key": self.deepseek_api_key,
                "model": self.deepseek_model,
                **base_config
            },
            "alibaba": {
                "api_key": self.alibaba_api_key,
                "model": self.alibaba_model,
                **base_config
            },
            "meta": {
                "api_key": self.meta_api_key,
                "model": self.meta_model,
                **base_config
            },
            "moonshot": {
                "api_key": self.moonshot_api_key,
                "model": self.moonshot_model,
                **base_config
            },
            "playai": {
                "api_key": self.playai_api_key,
                "model": self.playai_model,
                **base_config
            }
        }

        return provider_configs.get(provider_name, provider_configs["groq"])


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra='ignore')

    database_url: Optional[str] = Field(None, alias="DATABASE_URL")
    redis_url: Optional[str] = Field(None, alias="REDIS_URL")


class APIConfig(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra='ignore')

    host: str = Field("0.0.0.0", alias="API_HOST")
    port: int = Field(8000, alias="API_PORT")
    workers: int = Field(4, alias="API_WORKERS")


class SecurityConfig(BaseSettings):
    """Security configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra='ignore')

    secret_key: str = Field("healthcare_system_secret_key_2025", alias="SECRET_KEY")
    encryption_key: str = Field("healthcare_encryption_key_2025", alias="ENCRYPTION_KEY")


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra='ignore')

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_file: str = Field("logs/healthcare_system.log", alias="LOG_FILE")


class HealthcareSystemConfig:
    """Main configuration class that combines all settings."""

    def __init__(self):
        self.models = ModelProviderConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()

        # Add vision configuration for enhanced orchestrator
        self.vision = {
            "enabled": True,
            "provider": "google",
            "model": "openai/gpt-oss-20b",
            "max_image_size_mb": 10,
            "supported_formats": ["jpg", "jpeg", "png", "bmp"]
        }

    def get_provider_config(self, provider_name: str) -> Dict:
        """Get configuration for a specific provider."""
        return self.models.get_provider_config(provider_name)


# Global configuration instance
config = HealthcareSystemConfig()
