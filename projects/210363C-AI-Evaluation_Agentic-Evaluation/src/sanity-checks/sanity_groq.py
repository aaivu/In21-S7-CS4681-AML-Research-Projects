import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
assert api_key, "GROQ_API_KEY not set"

client = Groq(api_key=api_key)

resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",   
    messages=[{"role": "user", "content": "Say the api key works in one short line."}],
    temperature=0.2,
)
print(resp.choices[0].message.content)
