import torch
import time

def speculative_decode_kv(
    target_model,
    target_tokenizer,
    draft_model,
    draft_tokenizer,
    prompt,
    draft_steps=4,
    max_new_tokens=50
):

    input_ids = target_tokenizer(prompt, return_tensors="pt").input_ids.to(target_model.device)

    # ---- Build initial target cache ----
    with torch.no_grad():
        target_outputs = target_model(input_ids, use_cache=True)

    generated = input_ids
    past_target = target_outputs.past_key_values

    start = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # =====================================
            # 1. Draft proposes tokens
            # =====================================
            draft_ids = generated.clone()

            for _ in range(draft_steps):
                draft_outputs = draft_model(draft_ids)
                next_token = torch.argmax(draft_outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
                draft_ids = torch.cat([draft_ids, next_token], dim=-1)

            proposed = draft_ids[:, generated.shape[1]:]

            # =====================================
            # 2. Target verification (READ-ONLY)
            # =====================================
            verify_outputs = target_model(
                input_ids=proposed,
                past_key_values=past_target,
                use_cache=True
            )

            verify_logits = verify_outputs.logits

            accept_len = 0

            for i in range(proposed.shape[1]):
                predicted = torch.argmax(verify_logits[:, i, :], dim=-1)

                if predicted.item() == proposed[0, i].item():
                    accept_len += 1
                else:
                    break

            # =====================================
            # 3. Accept tokens safely
            # =====================================
            if accept_len > 0:

                accepted = proposed[:, :accept_len]

                # update sequence
                generated = torch.cat([generated, accepted], dim=-1)

                # IMPORTANT:
                # rebuild cache ONLY for accepted tokens
                accepted_outputs = target_model(
                    input_ids=accepted,
                    past_key_values=past_target,
                    use_cache=True
                )

                past_target = accepted_outputs.past_key_values

            else:
                # =================================
                # 4. Fallback single token
                # =================================
                fallback = target_model(
                    input_ids=generated[:, -1:],
                    past_key_values=past_target,
                    use_cache=True
                )

                next_token = torch.argmax(fallback.logits[:, -1, :], dim=-1).unsqueeze(-1)

                generated = torch.cat([generated, next_token], dim=-1)
                past_target = fallback.past_key_values

    end = time.time()

    text = target_tokenizer.decode(generated[0], skip_special_tokens=True)

    return text, end - start
