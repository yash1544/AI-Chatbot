# ai_chatbot.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random

class AIChatBot:
    def __init__(self, model_name="gpt2", max_history=5, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.max_history = max_history
        self.chat_history = []

    def detect_intent(self, user_input):
        """Simple intent detection for better control."""
        greetings = ["hello", "hi", "hey", "namaste"]
        exits = ["bye", "exit", "quit", "goodbye"]

        text = user_input.lower()
        if any(word in text for word in greetings):
            return "greeting"
        if any(word in text for word in exits):
            return "exit"
        return "chat"

    def generate_response(self, user_input, max_new_tokens=100):
        # Add input to history
        self.chat_history.append(f"User: {user_input}")
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)

        # Combine conversation into a single string
        context = "\n".join(self.chat_history) + "\nBot:"
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=self.tokenizer.eos_token_id
        )

        reply = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract bot's response after the last "Bot:"
        bot_reply = reply.split("Bot:")[-1].strip()

        # Add bot response to history
        self.chat_history.append(f"Bot: {bot_reply}")
        return bot_reply

    def chat(self):
        print("🤖 Chatbot is ready! Type 'exit' to quit.\n")
        while True:
            user_input = input("You: ")
            intent = self.detect_intent(user_input)

            if intent == "exit":
                print("Bot: Goodbye! Have a great day!")
                break
            elif intent == "greeting":
                print("Bot: Hello! How can I assist you today?")
                continue

            response = self.generate_response(user_input)
            print(f"Bot: {response}")


if __name__ == "__main__":
    bot = AIChatBot(model_name="gpt2")  # or "distilgpt2" for smaller
    bot.chat()
