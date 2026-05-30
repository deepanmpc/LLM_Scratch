import re
from Tokenizer import Tokenizer

with open("/Users/deepandee/Desktop/LLM_Scratch/Tokens_data.txt",'r',encoding='utf-8') as f:
    RawText=f.read()

out=re.split(r'[.,!]|\s',RawText)
tokens=[]
for i in out:
    if i.strip():
        tokens.append(i)
words=sorted(set(tokens))
vocabulary={}
for id,token in enumerate(words):
    vocabulary[token]=id
    
tokenizer=Tokenizer(vocabulary)
print("\nTHE TOKEN IDS OF THE TEXT IS:\n\n")
text="""Chapter 10, Large Language Models, LLMs such as T5 and LLaMA are discussed in this chapter, with
fine-tuning and efficient inference as the main focus. The field of LLMs has made significant progress in
recent years with the development of models such as GPT-3 (175B), PaLM (540B), BLOOM (175B),
LlaMA (65B), Falcon (180B), and Mistral (7B)."""

ids=tokenizer.encoder(text)
print(ids)

print("\nTHE DECODED TEXT IS:\n\n")
texts=tokenizer.decoder(ids)
print(texts)