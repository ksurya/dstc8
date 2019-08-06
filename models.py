from allennlp.models import Model
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.nn.util import get_text_field_mask

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch


class UserIntentPredictor(Model):

    def __init__(self, vocab):
        super().__init__(vocab)
        # embedding layer for encoding text
        self.emb = PretrainedBertEmbedder("bert-base-uncased", requires_grad=False, top_layer_only=True)
        # layers
        self.l0 = nn.Linear(self.emb.output_dim, self.emb.output_dim)
        self.l1 = nn.Linear(self.emb.output_dim, self.emb.output_dim)
        self.l2 = nn.Linear(1, 1)

    def forward(self, turnid, batch):
        # encode utter [Batch, Tokens, Emb]
        utter = self.emb(batch["usr_utter"]["tokens"][:,turnid,:])
        utter = self.l0(utter)

        # encode intent desc [Batch, Service, Intent, Tokens, Emb]
        intent_desc = self.emb(batch["intent_desc"]["tokens"])
        intent_desc = self.l1(intent_desc)

        # compute prediction score
        # see https://github.com/allenai/allennlp/issues/2668
        mask_usr_utter = (batch["usr_utter"]["tokens"][:,turnid,:] != 0).float()
        mask_intent_desc = (batch["intent_desc"]["tokens"] != 0).float()
        
        score = torch.einsum(
            "bxe,bsiye,bx,bsiy->bsi", 
            utter, intent_desc, mask_usr_utter, mask_intent_desc
        ) # [B, S, I]
        score = self.l2(score.unsqueeze(-1)).squeeze(-1) # [B, S, I]
        score = torch.sigmoid(score)

        output = {"score": score}
        if "intent_exist" in batch:
            target_score = batch["intent_exist"][:, turnid, ...] # [Batch, Service, Intent]
            output["loss"] = F.binary_cross_entropy(score, target_score)

        return output
