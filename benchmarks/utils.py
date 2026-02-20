def make_prompt(base_text, target_tokens, tokenizer):
    tokens = tokenizer(base_text).input_ids

    while len(tokens) < target_tokens:
        tokens += tokens

    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)
