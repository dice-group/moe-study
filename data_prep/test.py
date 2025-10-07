from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", use_fast=True)
print("unk_token:", tokenizer.unk_token)
print("unk_token_id:", tokenizer.unk_token_id)