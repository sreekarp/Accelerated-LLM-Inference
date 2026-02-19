import torch
import time

def speculative_decode(
    target_model,
    target_tokenizer,
    draft_model,
    draft_tokenizer,
    prompt,
    draft_steps=4,
    max_new_tokens=50
):

    input_ids = target_tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)
    generated = input_ids

    start = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # ---- Draft model proposes tokens ----
            draft_ids = generated.clone()

            for _ in range(draft_steps):
                draft_outputs = draft_model(draft_ids)
                draft_next = torch.argmax(draft_outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                draft_ids = torch.cat([draft_ids, draft_next], dim=-1)

            proposed_tokens = draft_ids[:, generated.shape[1]:]

            # ---- Target verifies ----
            outputs = target_model(draft_ids)
            logits = outputs.logits

            accept_length = 0

            for i in range(proposed_tokens.shape[1]):
                target_next = torch.argmax(logits[:, generated.shape[1] + i - 1, :], dim=-1)

                if target_next.item() == proposed_tokens[0, i].item():
                    accept_length += 1
                else:
                    break

            # ---- Accept tokens ----
            if accept_length > 0:
                generated = torch.cat([generated, proposed_tokens[:, :accept_length]], dim=-1)
            else:
                # fallback to normal decoding
                outputs = target_model(generated)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                generated = torch.cat([generated, next_token], dim=-1)

    end = time.time()

    text = target_tokenizer.decode(generated[0], skip_special_tokens=True)

    return text, end - start
