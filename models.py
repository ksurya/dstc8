from allennlp.models import Model
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch


class DialogState(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.l0 = nn.GRU(emb_size, emb_size, batch_first=True)

    def _get_state(self, deviceid, reset):
        # obj is not threadsafe, so we need diff obj for each device
        obj_name = f"state_{deviceid}"
        if reset or obj_name not in self.__dict__:
            setattr(self, obj_name, dict(l0=None))
        state = getattr(self, obj_name)
        if state["l0"] is not None:
            state["l0"] = state["l0"].detach()
        return state

    def forward(self, utter, desc, reset=False):
        assert utter.device.index == desc.device.index
        state = self._get_state(utter.device.index, reset)

        # encode utter
        ht, hw = self.l0(utter, state["l0"])
        state["l0"] = hw
        
        # [batch, emb]
        context = hw
        return context


class UserIntentPredictor(Model):

    def __init__(self, vocab):
        super().__init__(vocab)
        # embedding layer for encoding text
        self.emb = PretrainedBertEmbedder("bert-base-uncased", requires_grad=False, top_layer_only=True)
        
        # encode utter and desc tokens
        self.l0 = nn.GRU(self.emb.output_dim, self.emb.output_dim, batch_first=True)
        self.l1 = nn.GRU(self.emb.output_dim, self.emb.output_dim, batch_first=True)

        # maintain dialog state
        self.l2 = DialogState(self.emb.output_dim)

        # score
        #self.l3 = nn.Linear(self.emb.output_dim, 1)
        
        # metrics
        self.accuracy = CategoricalAccuracy()

        # init weights
        # TODO: classifier's performance changes heavily 
        # when weights are set to 0 / normal dist
        # when bias is set 0
        for name, param in self.named_parameters():
            if not name.startswith("emb."):
                if "weight" in name:
                    print("Initializing weights of ", name)
                    nn.init.xavier_uniform_(param)

    def get_metrics(self, reset):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def _get_score(self, ht_utter, hw_utter, ht_desc, hw_desc, context):
        hw = hw_desc * hw_utter * context
        mat = torch.einsum("lbe,bite->bi", hw, ht_desc)
        mat = torch.sigmoid(mat)
        return mat

    def forward(self, **batch):
        turnid = int(batch["turnid"])
        serviceid = int(batch["serviceid"])

        # tensors to play with
        utter = batch["usr_utter"]["tokens"][:,turnid,:] # [batch, tokens]
        utter = self.emb(utter) # [batch, tokens, emb]
        ht_utter, hw_utter = self.l0(utter) # [batch, tokens, emb * dir] and [layers * dir, batch, emb]

        intent_desc = batch["intent_desc"]["tokens"] # [batch, intents, tokens]
        intent_desc = self.emb(intent_desc)[:,serviceid,:]  # [batch, intents, tokens, emb] slice after embedding.. not working otherwise
        ht_desc, hw_desc = self.l1(torch.flatten(intent_desc, 1, 2)) # [batch, tokens, emb * dir] and [layers * dim, batch, emb]
        ht_desc = ht_desc.reshape(list(intent_desc.shape[:-1]) + [ht_desc.shape[-1]]) # [batch, intents, tokens, emb * dir]

        # context
        context = self.l2(utter, intent_desc, reset=turnid==0) # [layers * dim, batch, emb]

        # compute prediction onehot
        score = self._get_score(
            ht_utter,
            hw_utter,
            ht_desc,
            hw_desc,
            context,
        )

        output = {"score": score}

        if "intent_exist" in batch:
            # target values
            target_score = batch["intent_exist"][:,turnid,serviceid,:] # [Batch, Intent]
            values, target_labels = torch.max(target_score, -1) # [Batch,]

            # update the accuracy counters. reset at every dialogue
            mask = (values != -1).float()
            self.accuracy(
                score.float(),
                target_labels.float(), 
                mask=mask,
            )

            # print("Target", target_score.tolist())
            # print("Pred: ", score.tolist())
            # print("Mask: ", mask.tolist())
            # print("Acc: ", self.accuracy.get_metric())
            # print("\n\n")

            # calculate loss
            mask = (target_score != -1).float()
            output["loss"] = F.binary_cross_entropy(
                score.float(),
                target_score.float(),
                weight=mask,
                #reduction="sum"
            )

            #output["loss"] /= target_score.shape[0]

        return output
