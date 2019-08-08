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
        for batch in self.iterator(*args, **kw):
            num_turns = batch["usr_utter"]["tokens"].shape[1]
            num_services = batch["service_desc"]["tokens"].shape[1]
            for turnid in range(num_turns):
                for sid in range(num_services):
                    inputs = dict(turnid=turnid, serviceid=sid)
                    inputs.update(batch)
                    yield inputs


if __name__ == "__main__":
    # $ CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u train.py &
    from allennlp.data.iterators import BasicIterator
    from allennlp.training import Trainer
    from models import UserIntentPredictor
    from readers import DialogueReader
    import os

    allen_device = [int(i) for i in os.environ.get("CUDA_VISIBLE_DEVICES", "-1").split(",")]
    torch_device = "cuda" if allen_device[0] != -1 else "cpu"

    print("loading training data")
    train_ds = torch.load("data/preprocessed/dataset.pkl")
    vocab = torch.load("data/preprocessed/vocab.pkl")

    print("loading model")
    model = UserIntentPredictor(vocab).to(torch_device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    iterator = BasicIterator(batch_size=32)
    iterator.index_with(vocab)

    print("started training")
    trainer = Trainer(
        model=model,
        optimizer=optim,
        iterator=DialogIterator(iterator),
        train_dataset=train_ds,
        num_epochs=2,
        cuda_device=allen_device,
    )

    trainer.train()
