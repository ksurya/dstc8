from tqdm import tqdm_notebook as tqdm
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.training.trainer import Trainer

from readers import DialogueReader

# read data
reader = DialogueReader(limit=1)
train_dataset = reader.read("data/train") 

# build the vocab
vocab = Vocabulary.from_instances(train_dataset)

# litmus test
iterator = BasicIterator(batch_size=5)
iterator.index_with(vocab)
batch = next(iter(iterator(train_dataset)))
