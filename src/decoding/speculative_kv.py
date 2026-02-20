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
    generated = input_ids

    # 1. Initialize caches with the prompt
    with torch.no_grad():
        target_outputs = target_model(input_ids, use_cache=True)
        draft_outputs = draft_model(input_ids, use_cache=True)
        
    past_target = target_outputs.past_key_values
    past_draft = draft_outputs.past_key_values
    
    # The very first token is generated purely by the target model
    next_token = torch.argmax(target_outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated = torch.cat([generated, next_token], dim=-1)

    start = time.time()
    num_generated = 1

    with torch.no_grad():
        while num_generated < max_new_tokens:
            
            # =====================================
            # Phase 1: Draft Phase (O(1) Cached)
            # =====================================
            proposed = []
            draft_input = next_token
            
            for _ in range(draft_steps):
                draft_out = draft_model(
                    input_ids=draft_input, 
                    past_key_values=past_draft, 
                    use_cache=True
                )
                past_draft = draft_out.past_key_values
                draft_next = torch.argmax(draft_out.logits[:, -1, :], dim=-1).unsqueeze(-1)
                proposed.append(draft_next)
                draft_input = draft_next
                
            proposed_tensor = torch.cat(proposed, dim=-1)
            
            # =====================================
            # Phase 2: Target Verification 
            # (Only ONE target pass for K+1 tokens)
            # =====================================
            # verify_input length is K + 1 (next_token + proposed tokens)
            verify_input = torch.cat([next_token, proposed_tensor], dim=-1)
            
            verify_outputs = target_model(
                input_ids=verify_input,
                past_key_values=past_target,
                use_cache=True
            )
            past_target = verify_outputs.past_key_values
            
            # verify_preds gives us expected alignments for all drafted tokens + fallback
            verify_preds = torch.argmax(verify_outputs.logits, dim=-1)
            
            accept_len = 0
            for i in range(draft_steps):
                if proposed_tensor[0, i].item() == verify_preds[0, i].item():
                    accept_len += 1
                else:
                    break
            
            # =====================================
            # Phase 3: Zero-Overhead Cache Cropping
            # =====================================
            fallback_token = verify_preds[:, accept_len].unsqueeze(-1)
            
            if accept_len > 0:
                valid_new_tokens = torch.cat([proposed_tensor[:, :accept_len], fallback_token], dim=-1)
            else:
                valid_new_tokens = fallback_token
            
            # The length of cache we want to KEEP mathematically aligns with (old_sequence_len + accept_len)
            keep_len = generated.shape[1] + accept_len
            
            # Stop cleanly if the model outputs an EOS token inside the newly validated chunk
            eos_positions = (valid_new_tokens == target_tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
            if len(eos_positions) > 0:
                valid_new_tokens = valid_new_tokens[:, :eos_positions[0]+1]
                generated = torch.cat([generated, valid_new_tokens], dim=-1)
                break
                
            generated = torch.cat([generated, valid_new_tokens], dim=-1)
            num_generated += valid_new_tokens.shape[1]
            
            # If all proposed tokens were accepted, the draft model cache naturally missed `p_K`
            # We briefly run it to sync its cache to perfectly match target cache length
            if accept_len == draft_steps:
                draft_out = draft_model(
                    input_ids=proposed_tensor[:, -1:], 
                    past_key_values=past_draft,
                    use_cache=True
                )
                past_draft = draft_out.past_key_values
            
            # Safely crop caches natively via HF DynamicCache. No manual Tensor manipulation needed!
            if hasattr(past_target, "crop"):
                past_target.crop(keep_len)
                past_draft.crop(keep_len)
            else:
                past_target = tuple((k[:, :, :keep_len, :], v[:, :, :keep_len, :]) for k, v in past_target)
                past_draft = tuple((k[:, :, :keep_len, :], v[:, :, :keep_len, :]) for k, v in past_draft)
            
            # Setup the next_token pointer exactly
            next_token = fallback_token

    end = time.time()
    text = target_tokenizer.decode(generated[0], skip_special_tokens=True)

    return text, end - start