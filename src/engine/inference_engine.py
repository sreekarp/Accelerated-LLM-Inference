from src.decoding.greedy import greedy_decode
from src.decoding.kv_cache import kv_cache_decode
from src.decoding.speculative import speculative_decode

class InferenceEngine:
    """
    Core engine for LLM inference.
    All decoding methods will be added here.
    """

    def __init__(self, model, tokenizer, draft_model, draft_tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
    # ---- Baseline Greedy Decoding ----
    def generate_greedy(self, prompt, max_new_tokens=50):
        return greedy_decode(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens
        )

    # ---- KV Cache ----
    def generate_with_kv_cache(self, prompt, max_new_tokens=50):
        return kv_cache_decode(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens
        )
    # ---- Speculative Decoding ----
    def generate_speculative(self, prompt):
        return speculative_decode(
            self.model,
            self.tokenizer,
            self.draft_model,
            self.draft_tokenizer,
            prompt
        )

