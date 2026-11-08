
import os
from dotenv import load_dotenv

load_dotenv()

# API credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Model configuration
OPENAI_MODEL = "gpt-6.1-terra"
NVIDIA_MODEL = "nvidia/Nemotron-3.5-Super-VL"
JUDGE_MODEL = "gpt-6.1-terra"

NVIDIA_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

# Inference configuration
TEMPERATURE = 0
MAX_RETRIES = 1
REQUEST_TIMEOUT = 180
MAX_OUTPUT_TOKENS = 8192

# Dataset configuration
TOTAL_TASKS = 10

# Scoring configuration
ABSOLUTE_TOLERANCE = 0.01
RELATIVE_TOLERANCE = 0.01

# Validate credentials
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY missing from .env")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY missing from .env")

print("CircuitMind-Eval configuration loaded successfully!")
print(f"OpenAI model: {OPENAI_MODEL}")
print(f"Nebius model: {NVIDIA_MODEL}")
