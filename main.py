from tqdm import tqdm_notebook as tqdm
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer

from readers import DialogueDatasetReader
from models import UserIntentPredictor

import torch.optim as optim

# read data
reader = DialogueDatasetReader(limit=5)
train_dataset = reader.read("data/train") 

# build the vocab
vocab = Vocabulary.from_instances(train_dataset)

# litmus test
iterator = BasicIterator(batch_size=5)
iterator.index_with(vocab)
batch = next(iter(iterator(train_dataset)))

# model
model = UserIntentPredictor(vocab).to("cuda")

optimizer = optim.SGD(model.parameters(), lr=0.0001)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_dataset,
    num_epochs=2,
    cuda_device=0
)

trainer.train()
