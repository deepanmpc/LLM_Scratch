import re
from Tokenizers.Tokenizerv1 import TokenizerV1
from Tokenizers.Tokenizerv2 import TokenizerV2
from Tokenizers.BPE_Tokenizer import BPE_Tokenizer

with open(
    "/Users/deepandee/Desktop/LLM_Scratch/Data/Tokens_data.txt", "r", encoding="utf-8"
) as f:
    RawText = f.read()

out = re.split(r"[.,!]|\s", RawText)
tokens = []
for i in out:
    if i.strip():
        tokens.append(i)
words = sorted(set(tokens))

"""the vocabulary is mapping all the unique tokens to
 id and also 2 extra tokens to handle the unknown tokens and the end of text token"""

words.extend(["<|EOS|>", "<|unk|>", "<|BOS|>"])
vocabulary = {}
for id, token in enumerate(words):
    vocabulary[token] = id


"""This is to text and find the length of BPE okeniser results, it reduced from 700k to 200k tokens"""
BPE_tokenizer = BPE_Tokenizer("gpt2")
BPE_tokens = BPE_tokenizer.encoder(RawText)
BPE_words = [BPE_tokenizer.decoder([token]) for token in BPE_tokens]
print(BPE_words[:100])
print(len(BPE_tokens))
print(BPE_tokens[:100])


"""tokenizer = TokenizerV1(vocabulary)
tokenizerv2 = TokenizerV2(vocabulary)
print("\nTHE TOKEN IDS OF THE TEXT IS:\n\n")"""


text = """Chapter 10, Large Language Models, LLMs such as T5 and LLaMA are discussed in this chapter, with
fine-tuning and efficient inference as the main focus. The field of LLMs has made significant progress in
recent years with the development of models such as GPT-3 (175B), PaLM (540B), BLOOM (175B),
LlaMA (65B), Falcon (180B), and Mistral (7B)."""

#edge_case_texts = """Large Language Models,HI THIS TXT IS NOT FROM TRAINING DATA FROM THE VOCABULARY"""

"""ids = tokenizer.encoder(text)
print(ids)

print("\nTHE DECODED TEXT IS:\n\n")
texts = tokenizer.decoder(ids)
print(texts)

print("\nNEXT IS GONNA BE HANDLED BY V2 TOKENIZER\n")
print("\n THE TOKEN IDS OF THE EDGE CASE TEXT IS:\n\n")
edge_ids = tokenizerv2.encoder(edge_case_texts)
print(edge_ids)


print("\nTHE DECODED EDGE CASE TEXT IS:\n\n")
edge_texts = tokenizerv2.decoder(edge_ids)
print(edge_texts)"""
