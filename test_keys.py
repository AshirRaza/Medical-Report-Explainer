from dotenv import load_dotenv
import os

load_dotenv()

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")
qwen_model = os.getenv("QWEN_VL_MODEL")
claude_model = os.getenv("CLAUDE_MODEL")

print(f"Anthropic key: {'loaded' if anthropic_key else 'MISSING'}")
print(f"OpenRouter key: {'loaded' if openrouter_key else 'MISSING'}")
print(f"Qwen model: {qwen_model}")
print(f"Claude model: {claude_model}")
