## Encoding and decoding function
def encode_vocab(text, vocab):
    return [vocab.index(x) for x in text if x in vocab]

def decode_vocab(array, vocab):
    return ''.join([vocab[x - 1] for x in array])