from allennlp.models import Model
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import BooleanAccuracy

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch



class StatefulMultiheadAttention(nn.MultiheadAttention):

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.saved_state = {}

    def _get_input_buffer(self, incremental_state):
        return self.saved_state

    def _set_input_buffer(self, incremental_state, saved_state):
        self.saved_state = saved_state

    def reset_input_buffer(self):
        self.saved_state = {}


class DialogContext(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        # https://arxiv.org/pdf/1706.03762.pdf
        self.l0 = StatefulMultiheadAttention(emb_size, 2)

    def reset_input_buffer(self):
        self.l0.reset_input_buffer()

    def forward(self, utter_emb, desc_emb):
        # query [Tokens, Batch, Emb]
        utter_emb = utter_emb.permute(1,0,2)

        # key and values [Service * Tokens, Batch, Emb]
        desc_emb = torch.flatten(desc_emb, 1, 2)
        desc_emb = desc_emb.permute(1,0,2)

        # output [Tokens, Batch, Emb]
        # weights [Batch, Tokens, Service * Tokens]
        output, weights = self.l0(
            utter_emb, 
            desc_emb, 
            desc_emb, 
            incremental_state=True,
            need_weights=True,
        )

        # change shapes
        output = output.permute(1,0,2)

        return output


class UserIntentPredictor(Model):

    def __init__(self, vocab):
        super().__init__(vocab)
        # embedding layer for encoding text
        self.emb = PretrainedBertEmbedder("bert-base-uncased", requires_grad=False, top_layer_only=True)
        # layers
        self.l0 = nn.Linear(self.emb.output_dim, self.emb.output_dim)
        self.l1 = nn.Linear(self.emb.output_dim, self.emb.output_dim)
        #self.l2 = DialogContext(self.emb.output_dim)
        # metrics
        self.accuracy = BooleanAccuracy()

        # init weights
        for name, param in self.named_parameters():
            if not name.startswith("emb."):
                param.data.fill_(0)

    def reset_context(self):
        self.l2.reset_input_buffer()

    def get_metrics(self, reset):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def forward(self, **batch):
        turnid = int(batch["turnid"])
        serviceid = int(batch["serviceid"])

        # encode utter [Batch, Tokens, Emb]
        utter = self.emb(batch["usr_utter"]["tokens"][:,turnid,:])
        utter = self.l0(utter)

        # encode intent desc [Batch, Intent, Tokens, Emb]
        intent_desc = self.emb(batch["intent_desc"]["tokens"])
        intent_desc = intent_desc[:,serviceid,:]
        intent_desc = self.l1(intent_desc)

        # [Batch, Tokens, Emb]
        #context = self.l2(utter, intent_desc)

        # see https://github.com/allenai/allennlp/issues/2668
        mask_utter = (batch["usr_utter"]["tokens"][:,turnid,:] != 0).float()
        mask_intent_desc = (batch["intent_desc"]["tokens"][:,serviceid,:] != 0).float()
        
        # ranking on intents at each service
        score = torch.einsum(
            "bxe,bx,biye,biy->bi", 
            utter, mask_utter,
            intent_desc, mask_intent_desc,
        )
        score = torch.sigmoid(score) # [B, I]
        labels = torch.argmax(score, -1) # [B,]
        output = {"score": score, "labels": labels}

        if "intent_exist" in batch:
            # target values
            target_score = batch["intent_exist"][:,turnid,serviceid,:] # [Batch, Intent]
            values, target_labels = torch.max(target_score, -1) # [Batch,]

            # update the accuracy counters. reset at every dialogue
            self.accuracy(
                labels.float(), 
                target_labels.float(), 
                mask=(values != -1).float(),
            )

            # calculate loss
            output["loss"] = F.binary_cross_entropy(
                score,
                target_score,
                weight=(target_score != -1).float(), # passing mask as weights
            )

        return output
    