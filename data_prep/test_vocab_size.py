from transformers import AutoTokenizer
""" helper to verfiy on how to properly extend vocabs"""
    
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Old vocab size:", len(old_tokenizer))

corpus = [["hello world", "this is a test", "foobar foobaz"]]
new_tokenizer = old_tokenizer.train_new_from_iterator(corpus, vocab_size=300)
print("\n[train_new_from_iterator] New vocab size:", len(new_tokenizer))

old_vocab = set(old_tokenizer.get_vocab().keys())
new_vocab = set(new_tokenizer.get_vocab().keys())

print("Shared tokens:", len(old_vocab & new_vocab))
print("Tokens only in old:", len(old_vocab - new_vocab))
print("Tokens only in new:", len(new_vocab - old_vocab))

extended_tokenizer = AutoTokenizer.from_pretrained("gpt2")
num_added = extended_tokenizer.add_tokens(["<NEW_TOKEN1>", "<NEW_TOKEN2>", "<FOOBAR>"])
print("\n[add_tokens] Added tokens:", num_added)
print("Extended vocab size:", len(extended_tokenizer))

extended_vocab = set(extended_tokenizer.get_vocab().keys())

print("Shared tokens with old:", len(old_vocab & extended_vocab))
print("Tokens only in extended:", len(extended_vocab - old_vocab))

print("\n=== Examples ===")
print("Unique to REPLACED:", list(new_vocab - old_vocab)[:10])
print("Unique to EXTENDED:", list(extended_vocab - old_vocab)[:10])
print("Shared OLD & REPLACED:", list(old_vocab & new_vocab)[:10])
print("Shared OLD & EXTENDED:", list(old_vocab & extended_vocab)[:10])
