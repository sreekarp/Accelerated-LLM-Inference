# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# import torch
# import os
# MODEL_NAME = "microsoft/phi-2"
# LOCAL_DIR = "./models_store/phi2"

# def load_target_model():
#     os.makedirs(LOCAL_DIR, exist_ok=True)
#     tokenizer = AutoTokenizer.from_pretrained(
#         MODEL_NAME,
#         cache_dir=LOCAL_DIR
#     )

#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         device_map="auto",
#         quantization_config=quant_config
#     )

#     model.eval()
#     return model, tokenizer

# def load_draft_model():
#     MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )

#     model.eval()
#     return model, tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os

TARGET_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR = "./models_store/qwen_target"

DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_DIR = "./models_store/qwen_draft"


def load_target_model():
    os.makedirs(TARGET_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL,
        cache_dir=TARGET_DIR
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        cache_dir=TARGET_DIR,            # ⭐ FIX
        device_map="auto",
        quantization_config=quant_config
    )

    model.eval()
    return model, tokenizer


def load_draft_model():
    os.makedirs(DRAFT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        DRAFT_MODEL,
        cache_dir=DRAFT_DIR
    )

    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL,
        cache_dir=DRAFT_DIR,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()
    return model, tokenizer
