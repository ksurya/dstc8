import torch
import math

from allennlp.common.util import is_lazy, ensure_list

class DialogIterator(object):
    """Unfold a batch of dialogues."""

    def __init__(self, iterator):
        self.iterator = iterator
    
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.iterator, attr)

    def get_num_batches(self, instances):
        # modified allennlp implementation
        if is_lazy(instances) and self._instances_per_epoch is None:
            return 1
        elif self._instances_per_epoch is not None:
            return math.ceil(self._instances_per_epoch / self._batch_size)
        else:
            batches = 0
            for dial in ensure_list(instances):
                batches += dial.fields["num_turns"].as_tensor({})
                batches += len(dial.fields["service"].metadata)
            return math.ceil(batches / self._batch_size)

    def __call__(self, *args, **kw):
        # in multiple devices, Trainer would not execute turn=0 at first in all devices!!
        for batch in self.iterator(*args, **kw):
            num_turns = batch["usr_utter"]["tokens"].shape[1]
            num_services = batch["service_desc"]["tokens"].shape[1]
            for turnid in range(num_turns):
                for sid in range(num_services):
                    inputs = dict(turnid=turnid, serviceid=sid)
                    inputs.update(batch)
                    yield inputs
            # for sid in range(num_services):
            #     for turnid in range(num_turns):
            #         inputs = dict(turnid=turnid, serviceid=sid)
            #         inputs.update(batch)
            #         yield inputs


if __name__ == "__main__":
    # $ CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train.py &
    from allennlp.data.iterators import BasicIterator
    from allennlp.training import Trainer
    from models import UserIntentPredictor
    from readers import DialogueReader
    import os

    this_dir = os.path.abspath(os.path.dirname(__file__))
    allen_device = 0
    torch_device = 0

    print("loading training data")
    train_ds = torch.load(os.path.join(this_dir, "data/preprocessed/train_ds.pkl"))
    vocab = torch.load(os.path.join(this_dir, "data/preprocessed/vocab.pkl"))

    print("loading model")
    model = UserIntentPredictor(vocab).to(torch_device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-6)
    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)
    iterator = DialogIterator(iterator)

    print("started training")
    trainer = Trainer(
        model=model,
        optimizer=optim,
        iterator=iterator,
        train_dataset=train_ds,
        num_epochs=2,
        cuda_device=allen_device,
    )

    trainer.train()
