from groq import Groq
from config import GROQ_API_KEY

# Initialize client using .env key
client = Groq(api_key=GROQ_API_KEY)
