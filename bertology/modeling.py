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
        self.mem_pos = nn.Embedding(100, config.hidden_size)
        self.turn_pos = nn.Embedding(50, config.hidden_size)

        # state encodings
        self.prev_state_enc = nn.Linear(config.hidden_size, 1)
        self.register_buffer("prev_state", None)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, position_ids=None, head_mask=None, turnids=None):

        # sequence positions
        pos = torch.zeros(input_ids.shape, device=input_ids.device).long()
        pos[torch.arange(pos.shape[0])] = torch.tensor(list(range(0, pos.shape[1])), device=input_ids.device).long()
        pos = pos * attention_mask.long()

        if position_ids is None:
            position_ids = pos

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        # positional encodings
        mem_pos_emb = self.mem_pos(pos * (1 - token_type_ids))
        turn_pos_emb = self.turn_pos(turnids.unsqueeze(-1))

        attn_enabled = False
        if (turnids == 0).all():
            self.prev_state = None
        if self.prev_state is None or self.prev_state.shape[0] != input_ids.shape[0]:
            state_attn = 0
        else:
            state_attn = F.softmax(self.prev_state_enc(self.prev_state), -1)
            attn_enabled = True
        sequence_output = sequence_output + mem_pos_emb + turn_pos_emb
        self.prev_state = sequence_output.detach()

        # if self.context_h is not None and self.context_h.shape[1] != sequence_output.shape[0]:
        #     self.context_h = None
        # sequence_output, context_h = self.dialog_context(sequence_output, self.context_h)
        # self.context_h = context_h.detach()

        if attn_enabled:
            logits = self.qa_outputs(sequence_output) * state_attn
        else:
            logits = self.qa_outputs(sequence_output)

        logits = logits.squeeze(-1)
        outputs = (logits, logits,) + outputs[2:]


        # logits = self.qa_outputs(sequence_output)
        # start_logits, end_logits = logits.split(1, dim=-1)
        # start_logits = start_logits.squeeze(-1)
        # end_logits = end_logits.squeeze(-1)
        # outputs = (start_logits, end_logits,) + outputs[2:]

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
            # start_loss = loss_fct(start_logits, start_positions)
            # end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2
            # outputs = (total_loss,) + outputs

            loss = loss_fct(logits, start_positions)
            outputs = (loss,) + outputs

        # (loss), start_logits, end_logits, (hidden_states), (attentions)
        return outputs
