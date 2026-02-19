import torch
import time

def greedy_decode(model, tokenizer, prompt, max_new_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    generated = input_ids
    start = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            if next_token.item() == tokenizer.eos_token_id:
                break # deals with the autoregressive nature of LLM, or else the statements will be not accurate.
            generated = torch.cat([generated, next_token], dim=-1)
       

    end = time.time()

    text = tokenizer.decode(generated[0], skip_special_tokens=True)

    return text, end - start
