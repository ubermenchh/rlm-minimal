"""
Gemini Client wrapper specifically for Gemini-3 models.
"""

import os
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("MODEL", "gemini-3-pro-preview")

def _convert_messages_for_gemini(messages: list[dict[str, str]]) -> list:
    gemini_messages = []
    system_content = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            system_content += content + "\n\n"
        elif role == "user":
            text_to_send = system_content + content if system_content else content
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": text_to_send}]
            })
            system_content = ""
        elif role == "assistant":
            gemini_messages.append({
                "role": "model",
                "parts": [{"text": content}]
            })

    return gemini_messages

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = MODEL):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

        # Implement cost tracking logic here.
    
    def completion(
        self,
        messages: list[dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        try:
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, dict):
                messages = [messages]

            response = self.client.models.generate_content(
                model=self.model,
                contents=_convert_messages_for_gemini(messages),
                config=types.GenerateContentConfig(max_output_tokens=max_tokens),
            )
            # return response.choices[0].message.content
            return response.text if response.text else ""

        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")
