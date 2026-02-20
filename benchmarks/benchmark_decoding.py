import time
import pandas as pd

from src.models.loader import load_target_model, load_draft_model
from src.engine.inference_engine import InferenceEngine
from benchmarks.utils import make_prompt


BASE_TEXT = """
Speculative decoding is an inference optimization technique
for large language models that accelerates generation by using
a smaller draft model to propose tokens which are verified by
a larger target model.
"""


PROMPT_LENGTHS = [75, 200, 400]


def run_benchmark():

    print("Loading models...")
    target_model, target_tokenizer = load_target_model()
    draft_model, draft_tokenizer = load_draft_model()

    engine = InferenceEngine(
        target_model, target_tokenizer,
        draft_model, draft_tokenizer
    )

    print("Generating prompts...")

    prompts = {
        length: make_prompt(BASE_TEXT, length, target_tokenizer)
        for length in PROMPT_LENGTHS
    }

    results = []

    for length, prompt in prompts.items():

        print(f"\n=== Prompt Length: {length} tokens ===")

        for mode in ["greedy", "kv", "spec_naive", "spec_kv"]:

            print(f"Running {mode}...")

            start = time.time()

            if mode == "greedy":
                _, latency = engine.generate_greedy(prompt)

            elif mode == "kv":
                _, latency = engine.generate_with_kv_cache(prompt)

            elif mode == "spec_naive":
                _, latency = engine.generate_speculative(prompt)

            else:
                _, latency = engine.generate_speculative_kv(prompt)

            results.append({
                "prompt_length": length,
                "mode": mode,
                "latency": latency
            })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)

    print("\nSaved results to benchmark_results.csv")
    print(df)


if __name__ == "__main__":
    run_benchmark()
