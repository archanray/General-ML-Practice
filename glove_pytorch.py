import torch
print("PyTorch Version: {}".format(torch.__version__))
import torchtext
print("Torch Text Version: {}".format(torchtext.__version__))
import numpy

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

# Load dataset and create data loaders
# load AG NEWS dataset and create dataloaders from it.
# load by calling AG_NEWS() from datasets module of torchtext
# returns train and test datasets separately
# set batch size to 1024

from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset

max_words = 25
embed_len = 300

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = [tokenizer(x) for x in X]
    X = [tokens+[""]*(max_words-len(tokens)) \
            if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    X_tensor = torch.zeros(len(batch), max_words, embed_len)
    for i, tokens in enumerate(X):
        X_tensor[i] = global_vectors.get_vecs_by_tokens(tokens)
    # subtract 1 from labels to have label indices starting in 0
    return X_tensor.reshape(len(batch), -1), torch.tensor(Y)-1

target_classes = ["World", "Sports", "Business", "Sci/Tech"]

train_dataset, test_dataset = torchtext.datasets.AG_NEWS()

train_dataset = to_map_style_dataset(train_dataset)
test_dataset = to_map_style_dataset(test_dataset)

train_loader =  DataLoader(train_dataset, batch_size=1024, \
    collate_fn=vectorize_batch)
test_loader =  DataLoader(test_dataset, batch_size=1024, \
    collate_fn=vectorize_batch)

# lets define the network
# 4 linear layers -> 256,128,64,4. Relu after each layer except the last layer

from torch import nn
from torch.nn import functional as F

class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingClassifier, self).__init__()
        self.layer1 = nn.Linear(max_words*embed_len, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64,4)

    def forward(self, X_batch):
        X_batch = F.relu(self.layer1(X_batch))
        X_batch = F.relu(self.layer2(X_batch))
        X_batch = F.relu(self.layer3(X_batch))
        X_batch = self.layer4(X_batch)
        return X_batch

from tqdm import tqdm
from sklearn.metrics import accuracy_score
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def CalcValLossAndAcc(model, loss_fn, val_loader):
    # assuming model is at the right device
    with torch.no_grad():
        Y_shuffled, Y_preds, losses = [], [], []
        for X, Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)

            preds = model(X)
            loss = loss_fn(preds, Y)
            losses.append(loss.item())

            Y_shuffled.append(Y)
            Y_preds.append(preds.argmax(dim=-1))

        Y_shuffled = torch.cat(Y_shuffled)
        Y_preds = torch.cat(Y_preds)

        print("Validation loss: {:.3f}".format(torch.tensor(losses).mean()))
        print("Validation accu: {:.3f}".format(
                            accuracy_score(Y_shuffled.detach().cpu().numpy(),
                                Y_preds.detach().cpu().numpy())
                            ))

def TrainModel(model, loss_fn, optimizer, train_loader, val_loader, epochs=10):
    model.to(device)
    for i in range(1, epochs+1):
        losses = []
        for X, Y in tqdm(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            Y_preds = model(X)

            loss = loss_fn(Y_preds, Y)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i%5 == 0:
            print("Train Loss : {:.3f}".format(torch.tensor(losses).mean()))
            CalcValLossAndAcc(model, loss_fn, val_loader)

from torch.optim import Adam
epochs = 25
learning_rate = 1e-3

loss_fn = nn.CrossEntropyLoss()
embed_classifier = EmbeddingClassifier()
optimizer = Adam(embed_classifier.parameters(), lr=learning_rate)

TrainModel(embed_classifier, loss_fn, optimizer, train_loader, test_loader, epochs)


def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)
    gc.collect()
    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)

    return Y_shuffled.detach().cpu().numpy(), \
            F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().cpu().numpy()

Y_actual, Y_preds = MakePredictions(embed_classifier, test_loader)

from sklearn.metrics import accuracy_score, classification_report, \
            confusion_matrix

print("Test Accuracy : {}".format(accuracy_score(Y_actual, Y_preds)))
print("\nClassification Report : ")
print(classification_report(Y_actual, Y_preds, target_names=target_classes))
print("\nConfusion Matrix : ")
print(confusion_matrix(Y_actual, Y_preds))

from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_confusion_matrix([target_classes[i] for i in Y_actual], 
                                    [target_classes[i] for i in Y_preds],
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Purples",
                                    hide_zeros=True,
                                    figsize=(5,5)
                                    );
plt.xticks(rotation=90);