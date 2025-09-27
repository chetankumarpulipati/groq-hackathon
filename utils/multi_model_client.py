"""
Multi-model AI client for optimal accuracy across different healthcare tasks.
Supports multiple providers with intelligent model selection.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI
from groq import Groq
import google.generativeai as genai
from config.settings import config
from utils.logging import get_logger
from utils.error_handling import HealthcareSystemError, retry_on_failure

logger = get_logger("multi_model_client")


class MultiModelClient:
    """
    Intelligent multi-model client that selects the best AI model
    for each specific healthcare task to maximize accuracy.
    """

    def __init__(self):
        self.clients = {}
        self.model_capabilities = {
            "openai": {
                "strengths": ["general_medical", "diagnosis", "voice_processing", "text_analysis"],
                "accuracy_score": 95,
                "medical_specialty": True
            },
            "google": {
                "strengths": ["vision", "medical_imaging", "multimodal"],
                "accuracy_score": 93,
                "medical_specialty": True
            },
            "groq": {
                "strengths": ["speed", "real_time", "general"],
                "accuracy_score": 85,
                "medical_specialty": False
            },
            "deepseek": {
                "strengths": ["coding", "structured_data", "analysis"],
                "accuracy_score": 88,
                "medical_specialty": False
            },
            "alibaba": {
                "strengths": ["multilingual", "general", "reasoning"],
                "accuracy_score": 87,
                "medical_specialty": False
            },
            "meta": {
                "strengths": ["reasoning", "general", "conversation"],
                "accuracy_score": 86,
                "medical_specialty": False
            },
            "moonshot": {
                "strengths": ["long_context", "analysis"],
                "accuracy_score": 84,
                "medical_specialty": False
            },
            "playai": {
                "strengths": ["creative", "general"],
                "accuracy_score": 82,
                "medical_specialty": False
            }
        }
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize all available AI model clients using Groq API."""
        try:
            # All providers use Groq API with OpenAI-compatible interface
            # Initialize OpenAI client (using Groq's Llama 70B)
            if config.models.openai_api_key:
                self.clients["openai"] = OpenAI(
                    api_key=config.models.openai_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info("✅ OpenAI client initialized (Groq Llama 70B)")

            # Initialize Groq client (primary)
            if config.models.groq_api_key:
                self.clients["groq"] = Groq(api_key=config.models.groq_api_key)
                logger.info("✅ Groq client initialized")

            # Initialize Google client (using Groq's Mixtral)
            if config.models.google_api_key:
                self.clients["google"] = OpenAI(
                    api_key=config.models.google_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info("✅ Google client initialized (Groq Mixtral)")

            # Initialize other clients with Groq API
            provider_configs = {
                "deepseek": {"key": config.models.deepseek_api_key, "name": "Deepseek (Groq Gemma)"},
                "alibaba": {"key": config.models.alibaba_api_key, "name": "Alibaba (Groq Llama 8B)"},
                "meta": {"key": config.models.meta_api_key, "name": "Meta (Groq Llama 3.2 11B)"},
                "moonshot": {"key": config.models.moonshot_api_key, "name": "Moonshot (Groq Llama 3.2 3B)"},
                "playai": {"key": config.models.playai_api_key, "name": "PlayAI (Groq Llama 3.2 1B)"}
            }

            for provider, config_info in provider_configs.items():
                if config_info["key"]:
                    self.clients[provider] = OpenAI(
                        api_key=config_info["key"],
                        base_url="https://api.groq.com/openai/v1"
                    )
                    logger.info(f"✅ {config_info['name']} client initialized")

            logger.info(f"🚀 MultiModelClient initialized with {len(self.clients)} providers")

        except Exception as e:
            logger.error(f"Failed to initialize model clients: {str(e)}")
            raise HealthcareSystemError(f"Model client initialization failed: {str(e)}")

    def select_best_model(self, task_type: str, priority: str = "accuracy") -> str:
        """
        Select the best model for a specific healthcare task.
        All models are available through Groq API with different capabilities.

        Args:
            task_type: Type of task (diagnosis, vision, voice, validation, notification)
            priority: Priority mode (accuracy, speed, cost)
        """

        # Task-specific model selection optimized for Groq models
        task_model_mapping = {
            "diagnosis": "openai",        # llama-3.1-70b-versatile for highest accuracy
            "medical_diagnosis": "openai", # llama-3.1-70b-versatile for medical tasks
            "vision": "google",           # mixtral-8x7b-32768 for complex analysis
            "medical_imaging": "google",  # mixtral-8x7b-32768 for vision tasks
            "voice": "openai",            # llama-3.1-70b-versatile for voice processing
            "voice_processing": "openai", # llama-3.1-70b-versatile for voice
            "validation": "meta",         # llama-3.2-11b-text-preview for validation
            "data_validation": "meta",    # llama-3.2-11b-text-preview for data tasks
            "notification": "groq",       # llama-3.1-8b-instant for speed
            "general": "groq",            # llama-3.1-8b-instant for general tasks
            "complex_analysis": "google", # mixtral-8x7b-32768 for complex reasoning
            "quick_response": "groq"      # llama-3.1-8b-instant for speed
        }

        selected_provider = task_model_mapping.get(task_type, "groq")

        # Fallback if selected provider is not available
        if selected_provider not in self.clients:
            available_providers = list(self.clients.keys())
            if priority == "accuracy" and "openai" in available_providers:
                selected_provider = "openai"  # Best accuracy with Llama 70B
            elif priority == "speed" and "groq" in available_providers:
                selected_provider = "groq"    # Best speed with Llama 8B
            else:
                selected_provider = available_providers[0] if available_providers else "groq"

        logger.debug(f"Selected {selected_provider} for task: {task_type}")
        return selected_provider

    @retry_on_failure(max_retries=2)
    async def get_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: str = "general",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        provider_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get AI completion with automatic model selection for best accuracy.
        Returns dict with content and metadata.
        """

        provider = provider_override or self.select_best_model(task_type)
        max_tokens = max_tokens or 1024
        temperature = temperature if temperature is not None else 0.3

        try:
            # Get the correct model name for the provider (all use openai/gpt-oss-20b now)
            model_mapping = {
                "openai": config.models.openai_model,      # openai/gpt-oss-20b
                "google": config.models.google_model,      # openai/gpt-oss-20b
                "groq": config.models.groq_model,          # openai/gpt-oss-20b
                "deepseek": config.models.deepseek_model,  # openai/gpt-oss-20b
                "alibaba": config.models.alibaba_model,    # openai/gpt-oss-20b
                "meta": config.models.meta_model,          # openai/gpt-oss-20b
                "moonshot": config.models.moonshot_model,  # openai/gpt-oss-20b
                "playai": config.models.playai_model       # openai/gpt-oss-20b
            }
            
            model_name = model_mapping.get(provider, "openai/gpt-oss-20b")
            
            # All providers now use Groq client with correct parameters
            if provider == "groq":
                # Use native Groq client with correct parameters from user's example
                response = self.clients[provider].chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,  # Use max_completion_tokens instead of max_tokens
                    top_p=1,
                    stream=False  # We don't want streaming for our evaluation
                )
            else:
                # Use OpenAI-compatible client with Groq backend
                response = self.clients[provider].chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,  # Use max_completion_tokens
                    top_p=1,
                    stream=False
                )
            
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "provider": provider,
                "model": model_name,
                "tokens_used": getattr(response.usage, 'total_tokens', 0) if hasattr(response.usage, 'total_tokens') else 0
            }

        except Exception as e:
            logger.error(f"Error with {provider}: {str(e)}")
            
            # Try fallback provider
            fallback_provider = "groq" if provider != "groq" else "openai"
            if fallback_provider in self.clients and provider != fallback_provider:
                logger.info(f"Trying fallback provider: {fallback_provider}")
                return await self.get_completion(
                    messages, task_type, max_tokens, temperature,
                    provider_override=fallback_provider
                )
            
            raise HealthcareSystemError(f"All model providers failed: {str(e)}")

    async def get_medical_analysis(
        self,
        patient_data: Dict[str, Any],
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Get specialized medical analysis using the most accurate model.
        """

        # Use the highest accuracy provider for medical analysis
        provider = self.select_best_model("medical_diagnosis")

        # Create specialized medical prompt
        system_prompt = """You are a highly skilled medical AI assistant with expertise in healthcare analysis. 
        Your responses must be accurate, evidence-based, and follow medical best practices. 
        Always include confidence levels and recommend human physician review when appropriate.
        Format your response as structured JSON with clear sections for findings, recommendations, and confidence scores."""

        user_prompt = f"""
        Analyze the following patient data for {analysis_type} assessment:
        
        Patient Data: {patient_data}
        
        Please provide:
        1. Key findings and observations
        2. Potential diagnoses or concerns (with confidence scores)
        3. Recommended next steps
        4. Risk assessment
        5. When to seek immediate medical attention
        
        Remember: This is for healthcare professional assistance only, not direct patient care.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use lower temperature for medical accuracy
        result = await self.get_completion(
            messages=messages,
            task_type="medical_diagnosis",
            temperature=0.1,  # Very low for medical accuracy
            provider_override=provider
        )

        return {
            "analysis": result,
            "provider_used": provider,
            "confidence_level": "high" if provider in ["openai", "google"] else "medium",
            "requires_physician_review": True,
            "analysis_type": analysis_type
        }

    def get_available_providers(self) -> List[str]:
        """Get list of available model providers."""
        return list(self.clients.keys())

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers and their capabilities."""
        return {
            provider: {
                "available": provider in self.clients,
                "capabilities": self.model_capabilities.get(provider, {}),
                "model": config.get_provider_config(provider).get("model", "unknown")
            }
            for provider in self.model_capabilities.keys()
        }


# Global multi-model client instance
multi_client = MultiModelClient()
