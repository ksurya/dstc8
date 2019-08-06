from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.nn.util import move_to_device

import torch
import logging

from models import UserIntentPredictor
from readers import DialogueReader


def to_model(obj):
    if isinstance(obj, torch.nn.DataParallel):
        return obj.module
    return obj


def trainer(model, optim, iterator, train_dataset, num_epochs, device, print_every):
    counter = 0
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.train()
    counter = 0
    for epoch in range(num_epochs):
        for batch in iterator(train_dataset):
            counter += 1
            batch = move_to_device(batch, 0)
            num_turns = batch["usr_utter"]["tokens"].shape[1]
            metrics = {"loss": 0, "acc": 0}
            for turnid in range(num_turns):
                optim.zero_grad()
                output = model(turnid, batch)
                output["loss"] = output["loss"].mean()
                output["loss"].backward()
                metrics["loss"] += output["loss"].item()
            metrics["acc"] = to_model(model).get_metrics(reset=True)["accuracy"]
            if counter % print_every == 0:
                print("Iteration: {}, Epoch: {}, Loss: {}, Acc: {}".format(
                    epoch, counter, metrics["loss"], metrics["acc"]))


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
        device="cuda",
        print_every=10
    )
