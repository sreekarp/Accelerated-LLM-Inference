from src.models.loader import load_target_model
from src.decoding.greedy import greedy_decode

prompt = "Explain speculative decoding in simple terms."

model, tokenizer = load_target_model()

text, latency = greedy_decode(model, tokenizer, prompt)

print("OUTPUT:\n", text)
print("\nLatency:", latency, "seconds")
