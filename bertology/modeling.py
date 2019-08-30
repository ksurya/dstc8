from pytorch_transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.nn import functional as F
import torch

class BertForMemory(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForMemory, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 1)

        # positional encodings
        self.mem_pos_a = nn.Embedding(384, config.hidden_size)
        self.mem_pos_b = nn.Embedding(384, config.hidden_size)
        self.turn_pos = nn.Embedding(50, config.hidden_size)

        # state encodings
        self.prev_state_enc = nn.Linear(config.hidden_size, config.hidden_size)
        self.register_buffer("prev_state", None)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, position_ids=None, head_mask=None, turnids=None, memids=None):

        # sequence positions
        pos = torch.zeros(input_ids.shape, device=input_ids.device).long()
        pos[torch.arange(pos.shape[0])] = torch.tensor(list(range(0, pos.shape[1])), device=input_ids.device).long()
        pos = pos * attention_mask.long()

        if position_ids is None:
            position_ids = pos

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        # positional encodings
        mem_pos_emb_a = self.mem_pos_a(memids)
        mem_pos_emb_b = self.mem_pos_a(pos)
        turn_pos_emb = self.turn_pos(turnids.unsqueeze(-1))

        if (turnids == 0).all():
            self.prev_state = None
        if self.prev_state is None or self.prev_state.shape[0] != input_ids.shape[0]:
            state = 0
        else:
            state = F.relu(self.prev_state_enc(self.prev_state))

        sequence_output = sequence_output + mem_pos_emb_a + mem_pos_emb_b + turn_pos_emb + state
        self.prev_state = sequence_output.detach()

        logits = self.qa_outputs(sequence_output)

        logits = logits.squeeze(-1)
        outputs = (logits, logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            loss = loss_fct(logits, start_positions)
            outputs = (loss,) + outputs

        # (loss), start_logits, end_logits, (hidden_states), (attentions)
        return outputs
