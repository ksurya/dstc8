from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.nn.util import move_to_device

import torch
import logging

from models import UserIntentPredictor
from readers import DialogueReader

def trainer(model, optim, iterator, train_dataset, num_epochs, cuda_device, print_every):
    counter = 0
    device = ("cuda" if torch.cuda.is_available()
              and cuda_device != -1 else "cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    for epoch in range(num_epochs):
        # train
        for batch in iterator(train_dataset):
            batch = move_to_device(batch, cuda_device)
            counter += 1
            turns = batch["usr_utter"]["tokens"].shape[1]
            dialog_loss = 0
            for turnid in range(turns):
                optim.zero_grad()
                output = model(turnid, batch)
                if isinstance(model, torch.nn.DataParallel):
                    output["loss"] = output["loss"].mean()
                output["loss"].backward()
                dialog_loss += output["loss"].item()
            if counter % print_every == 0:
                print("Iteration: {}, Epoch: {}, Loss: {}".format(
                    epoch, counter, dialog_loss))


if __name__ == "__main__":
    # $ CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train.py &
    print("loading training data")
    train_ds = torch.load("data/preprocessed/dataset.pkl")
    vocab = torch.load("data/preprocessed/vocab.pkl")

    print("loading model")
    model = UserIntentPredictor(vocab)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)

    print("started training")
    trainer(
        model=model,
        optim=optim,
        iterator=iterator,
        train_dataset=train_ds,
        num_epochs=10,
        cuda_device=0,
        print_every=10
    )
