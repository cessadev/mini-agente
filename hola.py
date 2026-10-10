import os
from google import genai

MODEL = os.getenv("MODEL", "gemini-3.8-flash")

client = genai.Client()
interaction = client.interactions.create(
    model=MODEL,
    input="Explica qué es un agente de IA en dos frases.",
)
print(interaction.output_text)