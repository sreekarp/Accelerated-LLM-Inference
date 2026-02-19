from src.models.loader import load_target_model, load_draft_model
from src.engine.inference_engine import InferenceEngine

print("Loading target model...")
target_model, target_tokenizer = load_target_model()
print("Loading draft model...")
draft_model, draft_tokenizer = load_draft_model()

#initializing inference engine
engine = InferenceEngine(target_model, target_tokenizer,draft_model, draft_tokenizer)

print("\nModel loaded.")
print("Type 'exit' to quit.")

while True:
    print("\nChoose decoding mode:")
    print("1 — Greedy (baseline)")
    print("2 — KV Cache (optimized)")
    print("3 — Speculative (coming later)")

    mode = input("Enter choice: ")

    if mode.lower() == "exit":
        break

    prompt = input("\nPrompt: ")

    if prompt.lower() == "exit":
        break

    # ---- Mode Selection ----
    if mode == "1":
        text, latency = engine.generate_greedy(prompt)

    elif mode == "2":
        text, latency = engine.generate_with_kv_cache(prompt)

    elif mode == "3":
        text, latency = engine.generate_speculative(prompt)

    else:
        print("Invalid choice.")
        continue

    print("\nOUTPUT:\n", text)
    print("\nLatency:", latency, "seconds")
