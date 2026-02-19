class StoppingCriteria:
    """
    Flexible stopping controller for generation.
    """

    def __init__(self, tokenizer, stop_sequences=None, max_new_tokens=None):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences or []
        self.max_new_tokens = max_new_tokens

    def should_stop(self, generated_ids, step_count):
        # Decode current text
        text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # ---- 1. Stop on sequences ----
        for seq in self.stop_sequences:
            if seq in text:
                return True

        # ---- 2. Stop on max tokens ----
        if self.max_new_tokens is not None and step_count >= self.max_new_tokens:
            return True

        return False
