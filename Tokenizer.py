import re

"""The tokenizer v1 can handle only the tokens
from the traning data, the tokenizer v2 can 
handle the unknown tokens as well by using the <|unk|> token 
and also the end of text token <|endoftext|> to indicate the end of the text"""

class TokenizerV1:
    def __init__(self, vocab):
        self.str_int = vocab
        self.int_str = {id: text for text, id in vocab.items()}

    def encoder(self, text):
        out = re.split(r"[.,!]|\s", text)
        tokens = []
        for i in out:
            if i.strip():
                tokens.append(i)
        ids = []
        for i in tokens:
            ids.append(self.str_int[i])
        return ids

    def decoder(self, ids):
        text = " ".join(self.int_str[i] for i in ids)
        text = re.sub(r"\s+([,.!])", r"\1", text)
        return text
