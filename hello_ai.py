import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from the .env file

if not os.getenv("GROQ_API_KEY"):
    raise SystemExit("GROQ_API_KEY not found. Copy .env.example to .env and paste your key.")

client = Groq()  # uses GROQ_API_KEY automatically

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "What is PM-Kisan scheme?"}],
)

print("Answer:", response.choices[0].message.content)
print(f"Tokens: input {response.usage.prompt_tokens}, output {response.usage.completion_tokens}")
