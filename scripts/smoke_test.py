"""Smoke test: one real call to the Gemini API, to check that the key and the model work.

    python3 scripts/smoke_test.py
    MODEL=gemini-3.5-flash-lite python3 scripts/smoke_test.py

It is NOT part of the test suite (it needs the network and a key, and uses quota).
"""
import os

from google import genai

MODEL = os.getenv("MODEL", "gemini-3.8-flash")

client = genai.Client()
interaction = client.interactions.create(
    model=MODEL,
    input="Explain what an AI agent is in two sentences.",
)
print(interaction.output_text)
