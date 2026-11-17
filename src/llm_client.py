"""Client for communicating with Google Gemini generative models."""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import requests

from src.config import GEMINI_MODELS_CASCADE, REQUEST_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


class GeminiClient:
    """Manages text generation using Google Gemini API with fallback cascade."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def generate(self, prompt: str) -> str:
        """Generates content for the provided prompt string.
        
        Args:
            prompt: Formatted instruction and context prompt.
            
        Returns:
            Generated response string.
            
        Raises:
            ValueError: If GEMINI_API_KEY is not configured.
            RuntimeError: If generation fails across all cascade models.
        """
        key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Please add it to your .env file or enter it in the web sidebar."
            )

        # 1. Direct REST API calls (fastest, most consistent across versions)
        for model in GEMINI_MODELS_CASCADE:
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model}:generateContent?key={key}"
                )
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                res = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
            except Exception as e:
                logger.debug("REST generation with model %s failed: %s", model, e)
                continue

        # 2. Try google-genai official SDK
        try:
            from google import genai
            client = genai.Client(api_key=key)
            for model in GEMINI_MODELS_CASCADE:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    if response and response.text:
                        return response.text
                except Exception as e:
                    logger.debug("SDK generation with model %s failed: %s", model, e)
                    continue
        except Exception:
            pass

        raise RuntimeError(
            "Failed to generate response with Gemini API. "
            "Please verify your internet connection and API key quota."
        )


def generate(prompt: str) -> str:
    """Module-level helper to generate response with default client."""
    client = GeminiClient()
    return client.generate(prompt)
