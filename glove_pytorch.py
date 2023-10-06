import torch
print("PyTorch Version: {}".format(torch.__version__))
import torchtext
print("Torch Text Version: {}".format(torchtext.__version__))

# Approach 1: GloVe `840B`
# create tokenizer
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")
sentence = "Hello, How are you?"
print(tokenizer(sentence))

# Load GloVe embeddings ---> this takes time during the first run
from torchtext.vocab import GloVe
global_vectors = GloVe(name="840B", dim=300)
embeddings = global_vectors.get_vecs_by_tokens\
				(tokenizer(sentence), lower_case_backup=True)
print(embeddings.shape)
# the following should print an all zeros vector
# global_vectors.get_vecs_by_tokens([""], lower_case_backup=True)