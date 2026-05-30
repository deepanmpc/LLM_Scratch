import re
"""This is supports SPECIAL CONTEXT TOKENS."""
"""This v2 tokenizer can handle the unknown tokens as well by using the <|unk|> 
token and also the end of text token to indicate the end of the text"""

class TokenizerV2:
    def __init__(self, vocab):
        self.str_int = vocab
        self.int_str = {id: text for text, id in vocab.items()}

    def encoder(self, text):
        out = re.split(r"[.,!]|\s", text)
        tokens = []
        tokens.append("<|BOS|>")

        for i in out:
            if i.strip():
                tokens.append(i)
        for i, token in enumerate(tokens):
            if token not in self.str_int:
                tokens[i] = "<|unk|>"
        tokens.append("<|EOS|>")
        
        ids = []
        for i in tokens:
            ids.append(self.str_int[i])
        return ids

    def decoder(self, ids):
        text = " ".join(self.int_str[i] for i in ids)
        text = re.sub(r"\s+([,.!])", r"\1", text)
        return text
