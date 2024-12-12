import openai
import threading
from typing import List

class ConversationContext:
    def __init__(self, model: str = "gpt-3.5-turbo", max_tokens: int = 4096):
        self.lock = threading.Lock()
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant and your name is rebecca. Maintain context of our ongoing conversation."
            }
        ]
        self.max_tokens = max_tokens
        self.model = model

    def add_message(self, role: str, content: str):
        with self.lock:
            self.messages.append({"role": role, "content": content})
            self.trim_context()

    def trim_context(self):
        while self.calculate_total_tokens() > self.max_tokens:
            # Keep the system message and start trimming from the oldest user/assistant messages
            if len(self.messages) > 1:
                self.messages.pop(1)  # Remove the oldest non-system message

    def calculate_total_tokens(self) -> int:
        # Estimate tokens based on characters
        tokens = 0
        for message in self.messages:
            tokens += len(message["content"]) // 4  # Rough estimate: 1 token ≈ 4 characters
        return tokens

    def get_messages(self) -> List[dict]:
        with self.lock:
            return self.messages

    def reset(self):
        with self.lock:
            self.messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Maintain context of our ongoing conversation."
                }
            ]
