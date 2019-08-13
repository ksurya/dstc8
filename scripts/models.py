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
        self.l0 = nn.GRU(emb_size, emb_size, num_layers=2, batch_first=True)

    def _get_state(self, deviceid, reset):
        # obj is not threadsafe, so we need diff obj for each device
        obj_name = f"state_{deviceid}"
        if reset or obj_name not in self.__dict__:
            setattr(self, obj_name, dict(l0=None))
        state = getattr(self, obj_name)
        if state["l0"] is not None:
            state["l0"] = state["l0"].detach()
        return state

    def forward(self, usr_utter, sys_utter, reset=False):
        state = self._get_state(usr_utter.device.index, reset)

        # encode utter
        ht_usr, hw = self.l0(usr_utter, state["l0"])
        if sys_utter is not None:
            ht_sys, hw = self.l0(sys_utter, hw)

        state["l0"] = hw
        return hw


class UserIntentPredictor(Model):

    def __init__(self, vocab):
        super().__init__(vocab)
        # embedding layer for encoding text
        self.emb = PretrainedBertEmbedder("bert-base-uncased", requires_grad=False, top_layer_only=True)
        
        # encode utter and desc tokens
        self.l0 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.emb.output_dim, 2), 2
        )

        self.l1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.emb.output_dim, 2), 2
        )
        
        # maintain dialog state
        self.l2 = DialogState(self.emb.output_dim)

        # score
        self.l3 = nn.Linear(self.emb.output_dim, 1)
        
        # metrics
        self.accuracy = CategoricalAccuracy()

        # init weights
        # TODO: classifier's performance changes heavily 
        # for name, param in self.named_parameters():
        #     if not name.startswith("emb.") and not name.startswith("l3."):
        #         print("Initializing bias/weights of ", name)
        #         if "weight" in name:
        #             nn.init.xavier_uniform_(param)
        #         else:
        #             param.data.fill_(0.)

    def get_metrics(self, reset):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def _get_score(self, usr_utter, sys_utter, intent_desc, context):
        mat = torch.einsum("bite,lbe->bie", intent_desc, context)
        mat = self.l3(mat).squeeze(-1)
        mat = F.softmax(mat, dim=-1)
        return mat

    def forward(self, **batch):
        turnid = int(batch["turnid"])
        serviceid = int(batch["serviceid"])

        usr_utter = batch["usr_utter"]["tokens"][:,turnid,:] # [batch, tokens]
        usr_utter = self.emb(usr_utter) # [batch, tokens, emb]
        usr_utter = self.l0(usr_utter.permute(1,0,2)).permute(1,0,2) # [batch, tokens, emb]

        sys_utter = None
        if turnid > 0:
            sys_utter = batch["sys_utter"]["tokens"][:,turnid-1,:] # [batch, tokens]
            sys_utter = self.emb(sys_utter) # [batch, tokens, emb]
            sys_utter = self.l0(sys_utter.permute(1,0,2)).permute(1,0,2) # [batch, tokens, emb]

        intent_desc = batch["intent_desc"]["tokens"] # [batch, intents, tokens]
        shape = intent_desc.shape
        intent_desc = self.emb(intent_desc)[:,serviceid,:]  # [batch, intents, tokens, emb] slice after embedding.. not working otherwise
        intent_desc = self.l1(torch.flatten(intent_desc, 1, 2).permute(1,0,2)).permute(1,0,2) # [batch, tokens, emb ]
        intent_desc = intent_desc.reshape(list(shape) + [-1]) # [batch, intents, tokens, emb]

        # context
        context = self.l2(usr_utter, sys_utter, reset=turnid==0) # [layers * dim, batch, emb]

        # compute prediction onehot
        score = self._get_score(
            usr_utter,
            sys_utter,
            intent_desc,
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
            
            with open("switch.txt") as f:
                switch = f.read()
                if switch == "1":
                    # print("Target", target_score.tolist())
                    # print("Pred: ", score.tolist())
                    print("Target", torch.argmax(target_score, -1).tolist())
                    print("Pred: ", torch.argmax(score, -1).tolist())
                    print("Mask: ", mask.long().tolist())
                    print("Acc: ", self.accuracy.get_metric())
                    print("\n\n")

            # calculate loss
            # mask = (target_score != -1).float()
            # output["loss"] = F.binary_cross_entropy(
            #     score.float(),
            #     target_score.float(),
            #     weight=mask,
            #     #reduction="sum"
            # )
            mask = (target_score != -1).float()
            output["loss"] = F.mse_loss(
                score * mask,
                target_score * mask,
                reduction="sum"
            )
            output["loss"] /= target_score.shape[0]

        return output
