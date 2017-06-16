from encode_decode import *

## Getting data from the filename
NUM_STEPS = 60
def read_data(filename, vocab, window=NUM_STEPS, overlap=NUM_STEPS/2):
    for text in open(filename):
        text = encode_vocab(text, vocab)
        for start in range(0, len(text) - window, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk