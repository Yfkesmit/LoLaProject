import string

def tokenize(text):
    tokens = [word.strip(string.punctuation) for word in text.split()]
    tokens = [token for token in tokens if token]
    return tokens