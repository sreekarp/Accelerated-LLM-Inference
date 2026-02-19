import torch
import time

def kv_cache_decode(model, tokenizer, prompt, max_new_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    generated = input_ids
    past = None

    start = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            outputs = model(
                input_ids=generated if past is None else next_token,
                past_key_values=past,
                use_cache=True
            )

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)

            generated = torch.cat([generated, next_token], dim=-1)

            # ⭐ SAVE CACHE
            past = outputs.past_key_values

    end = time.time()

    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return text, end - start
