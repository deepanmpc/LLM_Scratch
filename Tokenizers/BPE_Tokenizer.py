import tiktoken

class BPE_Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = tiktoken.get_encoding(model_name)

    def encoder(self, text):
        return self.tokenizer.encode(text)

    def decoder(self, tokens):
        return self.tokenizer.decode(tokens)
    
tokenizer=BPE_Tokenizer("gpt2")
text = "Hello, how are you?"
RandomText = "Taiugqfuief eifewfwe fwefiwefkwebf  rie"
tokens = tokenizer.encoder(text)
print("Tokens:", tokens)
detokenized_text = tokenizer.decoder(tokens)
print("Detokenized Text:", detokenized_text)


tokens = tokenizer.encoder(RandomText)
print("Tokens:", tokens)
detokenized_text = tokenizer.decoder(tokens)
print("Detokenized Text:", detokenized_text)