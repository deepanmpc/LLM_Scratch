

import re

with open("/Users/deepandee/Desktop/TOKENISER_LLM_ETC/Tokens_data.txt",'r',encoding='utf-8') as f:
    RawText=f.read()
import re
out=re.split(r'[.,!]|\s',RawText)
tokens=[]
for i in out:
    if i.strip():
        tokens.append(i)
words=sorted(set(tokens))
vocabulary={}
for id,token in enumerate(words):
    vocabulary[token]=id



class Tokenizer:
    def __init__(self,vocab):
        self.str_int=vocab
        self.int_str={id:text for text,id in vocab.items()}

    def encoder(self,text):
        out=re.split(r'[.,!]|\s',text)
        tokens=[]
        for i in out:
            if i.strip():
                tokens.append(i)
        ids=[]
        for i in tokens:
            ids.append(self.str_int[i])
        return ids
    def decoder(self, ids):
        text = " ".join(self.int_str[i] for i in ids)
        text = re.sub(r'\s+([,.!])', r'\1', text)
        return text
        