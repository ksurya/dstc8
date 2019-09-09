import os

import glob
import os
import json
import numpy as np
import re
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from bert_optim import Adamax, RAdam

from collections import OrderedDict, defaultdict
from functools import lru_cache
from ipdb import set_trace
from tqdm import tqdm
from tabulate import tabulate

from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, ListField, MetadataField, ArrayField, IndexField, Field, AdjacencyField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer, WordpieceIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.models import Model, SimpleSeq2Seq
from allennlp.modules.token_embedders import PretrainedBertEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertModel
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.training import Trainer
from allennlp.training.learning_rate_schedulers import SlantedTriangular, NoamLR, CosineWithRestarts
from allennlp.training.moving_average import ExponentialMovingAverage
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from allennlp.nn.util import get_text_field_mask, move_to_device


np.random.seed(1)


class Schema(object):

    def __init__(self, filepath):
        with open(filepath) as f:
            self.index = {}
            for schema in json.load(f):
                service_name = schema["service_name"]
                self.index[service_name] = schema

    @lru_cache(maxsize=None)
    def get(self, service):
        result = dict(
            # service
            name=service,
            desc=self.index[service]["description"],
            
            # slots
            slot_name=[],
            slot_desc=[],
            slot_iscat=[], 
            slot_vals=[], # collected only for cat slots.. not sure if that makes sense

            # intents
            intent_name=[],
            intent_desc=[],
            intent_istrans=[],
            intent_reqslots=[],
            intent_optslots=[],
            intent_optvals=[],
        )

        for slot in self.index[service]["slots"]:
            result["slot_name"].append(slot["name"])
            result["slot_desc"].append(slot["description"])
            result["slot_iscat"].append(slot["is_categorical"])
            result["slot_vals"].append(slot["possible_values"])
        
        for intent in self.index[service]["intents"]:
            result["intent_name"].append(intent["name"])
            result["intent_desc"].append(intent["description"])
            result["intent_istrans"].append(intent["is_transactional"])
            result["intent_reqslots"].append(intent["required_slots"])
            result["intent_optslots"].append(list(intent["optional_slots"].keys()))
            result["intent_optvals"].append(list(intent["optional_slots"].values()))

        return result


class Memory(object):
    
    def __init__(self, schema, services):
        self.schema = schema
        
        # memory for cat slots: serv,slot=>[] or for noncat slots: noncat => []
        self.memory = defaultdict(list)
        self.index = defaultdict(set) # serv,slot->val, noncat->val

        for serv in services:
            # add possible values from slot
            sch = schema.get(serv)
            for slot, iscat, slotvals in zip(sch["slot_name"], sch["slot_iscat"], sch["slot_vals"]):
                key = (serv, slot) if iscat else "noncat"
                for val in ["NONE", "dontcare"] + slotvals:
                    if val not in self.index[key]:
                        self.index[key].add(val)
                        self.memory[key].append(val)

            # add optional slot vals
            for optslots, optvals in zip(sch["intent_optslots"], sch["intent_optvals"]):
                for slot, val in zip(optslots, optvals):
                    slotid = sch["slot_name"].index(slot)
                    iscat = sch["slot_iscat"][slotid]
                    assert slotid != -1
                    key = (serv, slot) if iscat else "noncat"
                    if val not in self.index[key]:
                        self.index[key].add(val)
                        self.memory[key].append(val)
                        
    def update(self, dial_turn):
        # update only noncat values..
        utter = dial_turn["utterance"]
        for frame in dial_turn["frames"]:
            sch = self.schema.get(frame["service"])
            slot_names = sch["slot_name"]
            slot_iscat = sch["slot_iscat"]

            for tag in frame["slots"]:
                slot, st, en = tag["slot"], tag["start"], tag["exclusive_end"]
                slotid = slot_names.index(slot)
                iscat = slot_iscat[slotid]
                assert slotid != -1

                if not iscat:
                    value = utter[st:en]
                    key = "noncat"
                    if value not in self.index[key]:
                        value = re.sub("\u2013", "-", value) # dial 59_00125 turn 14
                        self.index[key].add(value)
                        self.memory[key].append(value)
                        
    def get(self, key="noncat"):
        return self.memory[key]
    
from typing import Dict, List, Callable
from pytorch_pretrained_bert.tokenization import BertTokenizer

class PretrainedBertIndexerCLS(WordpieceIndexer):

        def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True) -> None:

            bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase)
            super().__init__(vocab=bert_tokenizer.vocab,
                            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                            namespace="bert",
                            use_starting_offsets=use_starting_offsets,
                            max_pieces=max_pieces,
                            do_lowercase=do_lowercase,
                            never_lowercase=never_lowercase,
                            start_tokens=["[CLS]"],
                            end_tokens=[],
                            separator_token="[SEP]",
                            truncate_long_sequences=truncate_long_sequences)


class PretrainedBertIndexerSEP(WordpieceIndexer):

        def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True) -> None:

            bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase)
            super().__init__(vocab=bert_tokenizer.vocab,
                            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                            namespace="bert",
                            use_starting_offsets=use_starting_offsets,
                            max_pieces=max_pieces,
                            do_lowercase=do_lowercase,
                            never_lowercase=never_lowercase,
                            start_tokens=[],
                            end_tokens=["[SEP]"],
                            separator_token="[SEP]",
                            truncate_long_sequences=truncate_long_sequences)  

class PretrainedBertIndexerNoSep(WordpieceIndexer):

        def __init__(self,
                 pretrained_model: str,
                 use_starting_offsets: bool = False,
                 do_lowercase: bool = True,
                 never_lowercase: List[str] = None,
                 max_pieces: int = 512,
                 truncate_long_sequences: bool = True) -> None:

            bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=do_lowercase)
            super().__init__(vocab=bert_tokenizer.vocab,
                            wordpiece_tokenizer=bert_tokenizer.wordpiece_tokenizer.tokenize,
                            namespace="bert",
                            use_starting_offsets=use_starting_offsets,
                            max_pieces=max_pieces,
                            do_lowercase=do_lowercase,
                            never_lowercase=never_lowercase,
                            start_tokens=[],
                            end_tokens=[],
                            separator_token="[SEP]",
                            truncate_long_sequences=truncate_long_sequences)                                                  

class DialogReader(DatasetReader):

    def __init__(self, schema, limit, lazy=False):
        super().__init__(lazy)
        self.token_indexers = {"tokens": PretrainedBertIndexer("bert-base-uncased")}
        self.token_indexers_cls = {"tokens": PretrainedBertIndexerCLS("bert-base-uncased",)}
        self.token_indexers_sep = {"tokens": PretrainedBertIndexerSEP("bert-base-uncased",)}
        self.token_indexers_nosep = {"tokens": PretrainedBertIndexerNoSep("bert-base-uncased",)}
        self.tokenizer = BertBasicWordSplitter()
        self.schema = schema
        self.limit = limit

    def _read(self, path):
        # get a set of dialogs
        count = 0
        dialogs = []
        for filename in sorted(glob.glob(path)):
            if count > self.limit:
                break
            with open(filename) as f:
                for d in json.load(f):
                    dialogs.append(d)
                    count += 1
                    if count > self.limit:
                        break
        
        # prepare instances
        for dial in dialogs:
            memory = Memory(self.schema, dial["services"])
            for turnid, turn in enumerate(dial["turns"]):
                memory.update(turn)
                if turn["speaker"] == "USER":
                    usr_utter = turn["utterance"]
                    sys_utter = dial["turns"][turnid-1]["utterance"] if turnid > 0 else "dialog started"
                    num_none_questions  = 0
                    
                    for frame in turn["frames"]:
                        # get schema info
                        serv = frame["service"]
                        sch = self.schema.get(serv)
                        
                        # intent
                        intent = frame["state"]["active_intent"]
                        all_intents = {s: i for i, s in enumerate(sch["intent_name"])}
                        intent_istrans = False
                        intent_desc = "No intent"
                        if intent != "NONE":
                            intentid = all_intents[intent]
                            assert intentid != -1
                            intent_desc = sch["intent_desc"][intentid]
                            intent_istrans = sch["intent_istrans"][intentid]
                        
                        # slots
                        all_slots = {s: i for i, s in enumerate(sch["slot_name"])}
                        all_slots_iscat = sch["slot_iscat"]
                        all_slots_desc = sch["slot_desc"]
                        active_slots = frame["state"]["slot_values"]
                        none_slots = set(all_slots) - set(active_slots)
                        
                        # active slots
                        for slot, values in active_slots.items():
                            slotid = all_slots[slot]
                            assert slotid != -1
                            key = (serv, slot) if all_slots_iscat[slotid] else "noncat"
                            target_value = re.sub("\u2013", "-", values[0])
                            
                            item = dict(
                                dialid=dial["dialogue_id"],
                                turnid=turnid,
                                usr_utter=usr_utter,
                                sys_utter=sys_utter,
                                serv=serv,
                                serv_desc=sch["desc"],
                                slot=slot,
                                slot_desc=all_slots_desc[slotid],
                                slot_iscat=all_slots_iscat[slotid],
                                slot_val=target_value,
                                intent=intent,
                                intent_desc=intent_desc,
                                intent_istrans=intent_istrans,
                                memory=memory.get(key),
                            )
                            yield self.text_to_instance(item)
                            
            
    def text_to_instance(self, item):
        fields = {}
        
        # featurize query
        query_tokens = []
        query_type = []
        
        for index, field in enumerate(("sys_utter", "usr_utter", "serv_desc", "slot_desc")):
            tokens = self.tokenizer.split_words(item[field])
            query_tokens.extend(tokens)
            query_type.extend([index + 1] * len(tokens))
        
        query_pos = list(range(1, len(query_tokens) + 1))
        
        fields["query"] = TextField(query_tokens, self.token_indexers_sep)
        fields["query_type"] = ArrayField(np.array(query_type))
        fields["query_pos"] = ArrayField(np.array(query_pos))
        
        # featurize memory
        mem_values = item["memory"]
        mem_tokens = []
        mem_pos = []
        mem_loc = []
        mem_type = []
        
        for index, mem_val in enumerate(mem_values):
            tokens = self.tokenizer.split_words(mem_val)
            pos = np.array(range(1, len(tokens) + 1))
            mtype = np.array([index + 1] * len(tokens))
            istarget = int(mem_val == item["slot_val"])
            
            if index == 0:
                token_indexers = self.token_indexers_cls
            elif index == len(mem_values) - 1:
                token_indexers = self.token_indexers_sep
            else:
                token_indexers = self.token_indexers_nosep

            mem_tokens.append(TextField(tokens, token_indexers))
            mem_pos.append(ArrayField(pos))
            mem_type.append(ArrayField(mtype))
            mem_loc.append(istarget)
            
        fields["memory"] = ListField(mem_tokens)
        fields["memory_pos"] = ListField(mem_pos)
        fields["memory_type"] = ListField(mem_type)
        fields["memory_loc"] = ArrayField(np.array(mem_loc), padding_value=-1)

        # positional fields
        fields["turnid"] =  ArrayField(np.array(item["turnid"]))
        
        # meta fields
        fields["id"] = MetadataField("{}/{}/{}/{}".format(item["dialid"], item["turnid"], item["serv"], item["slot"]))
        fields["slot"] = MetadataField(item["slot"])
        fields["serv"] = MetadataField(item["serv"])
        fields["intent"] = MetadataField(item["intent"])
        fields["dialid"] = MetadataField(item["dialid"])
        fields["memory_values"] = MetadataField(item["memory"])
        
        return Instance(fields)


train_schema = Schema("../data/train/schema.json")
dev_schema = Schema("../data/dev/schema.json")


# read full dataset
reader = DialogReader(train_schema, limit=100)
train_ds = reader.read("../data/train/dialogues*.json")

reader = DialogReader(dev_schema, limit=10)
dev_ds = reader.read("../data/dev/dialogues*.json")

vocab = Vocabulary.from_instances(train_ds + dev_ds)


it = BasicIterator(batch_size=32)
it.index_with(vocab)
batch = next(iter(it(train_ds)))
batch.keys()


# shapes
for f in batch:
    if type(batch[f]) is torch.Tensor:
        print(f, "->", batch[f].shape)
    elif type(batch[f]) is dict and type(batch[f]["tokens"]) is torch.Tensor:
        print(f, "->", batch[f]["tokens"].shape)


class CandidateSelector(Model):
    
    def __init__(self, vocab):
        super().__init__(vocab)
        # query encoder
        self.emb = PretrainedBertEmbedder("/home/suryak/mtdnn/mt_dnn_models/pytorch_model.tar.gz", requires_grad=True)
        emb_dim = self.emb.get_output_dim()
        
        # memory decoder
        dec_layer = nn.TransformerDecoderLayer(emb_dim, 8)
        self.dec = nn.TransformerDecoder(dec_layer, 6)
        
        #self.enc_dec = nn.Transformer(emb_dim, emb_dim)
        
        # final
        self.final = nn.Linear(emb_dim, 1)
        
        # embeddings
#         self.query_type_emb = nn.Embedding(10, emb_dim) # max query fields
#         self.query_pos_emb = nn.Embedding(500, emb_dim) # max query length
#         self.memory_pos_emb = nn.Embedding(500, emb_dim) # max total length of candidates
#         self.memory_pos_emb = nn.Embedding(500, emb_dim) # max total length of candidates
        
        # metrics
        self.accuracy = BooleanAccuracy()
        
    def get_metrics(self, reset=False):
        return {"acc": self.accuracy.get_metric(reset)}
    
    def encoder(self, batch):
        q = batch["query"]
        m = batch["memory"]
        x_tokens = torch.cat((m["tokens"].flatten(1, 2), q["tokens"], ), -1) # b,s1+s2
        x_offsets = torch.cat((m["tokens-offsets"].flatten(1,2), q["tokens-offsets"]), -1)
        x_types = torch.cat((m["tokens-type-ids"].flatten(1,2), q["tokens-type-ids"] + 1), -1)
        
        enc = self.emb(x_tokens, x_offsets, x_types)
        
        m_len = q["mask"].shape[-1]
        tot = enc.shape[1]

        mem_enc, query_enc = enc.split([tot - m_len, m_len], 1)
        
        mem_sh = list(m["mask"].shape)
        mem_enc = mem_enc.view(mem_sh + [-1])
        
        return query_enc, mem_enc
    
    
    def decoder(self, query, memory):
        # query: encoder output, batch, seq, emb
        # memory: decoder input, batch, mem, seq, emb
        memory = memory.sum(2)
        return memory

        query = query.permute(1,0,2) # seq, batch, emb
        memory = memory.permute(1,0,2)
        
        x = self.dec(memory, query)
        #x = self.enc_dec(query, memory)
        x = x.permute(1,0,2)
        
        return x
    
    def mse_loss(self, decoded, target):
        # decoded: batch, mem, emb 
        # tgt: batch, mem
        predicted = F.softmax(self.final(decoded).squeeze(-1), -1) # batch, mem
        
        # loss
        mask = (target != -1).float()
        loss = F.mse_loss(predicted * mask, target * mask).unsqueeze(0)
        
        # metric
        predicted_loc = predicted.argmax(-1)
        target_loc = target.argmax(-1)

        mask_loc = mask[torch.arange(mask.shape[0]), target_loc]
        self.accuracy(predicted_loc, target_loc, mask_loc.long())
    
        return loss, predicted_loc, target_loc
    
    def bce_loss(self, decoded, target):
        # decoded: batch, mem, emb
        # tgt: batch, mem
        predicted = self.final(decoded).squeeze(-1)
        
        # loss
        loss = F.cross_entropy(predicted, target.argmax(-1), ignore_index=-1).unsqueeze(0)
        
        # metric
        predicted_loc = predicted.argmax(-1)
        target_loc = target.argmax(-1)

        mask = (target != -1).float()
        mask_loc = mask[torch.arange(mask.shape[0]), target_loc]
        self.accuracy(predicted_loc, target_loc, mask_loc.long())
        
        return loss, predicted_loc, target_loc
    
    def forward(self, **batch):
        query, memory = self.encoder(batch)
        #query = query.detach()
        #memory = memory.detach()

        decoded = self.decoder(query, memory) # batch, mem, emb

        target = batch["memory_loc"] # [batch, mem]
        loss, predicted_loc, target_loc = self.bce_loss(decoded, target)
        
        output = dict(
            loss=loss,
            pred=predicted_loc,
            target=target_loc,
        )
        
        return output


#%%
allen_device=2
torch_device=2

model = CandidateSelector(vocab).to(torch_device)
optimizer = optim.Adam(model.parameters(), lr=3e-5)
#optimizer.initialize_step(0)

iterator = BasicIterator(batch_size=16)
iterator.index_with(vocab)

num_steps = iterator.get_num_batches(train_ds)

moving_average = None #ExponentialMovingAverage(model.named_parameters())

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    num_epochs=3,
    cuda_device=allen_device,
    serialization_dir="../results/7",
    #should_log_learning_rate=True,
    #histogram_interval=1000,
    num_serialized_models_to_keep=1,
    grad_norm=5,
    shuffle=False,
    moving_average=moving_average,
)

trainer.train()


def predictor(model, test_ds, device):
    results = defaultdict(OrderedDict)
    test_iterator = BasicIterator(batch_size=32)
    test_iterator.index_with(vocab)
    
    model = move_to_device(model, device)
    model = model.eval()
    
    for sample in tqdm(test_iterator(test_ds, shuffle=False, num_epochs=1)):
        sample = move_to_device(sample, device)
        with torch.no_grad():
            output = model(**sample)
        
        num_samples = output["target"].shape[0]
        for i in range(num_samples):
            key = sample["id"][i]
            t_loc = int(output["target"][i].item())
            p_loc = int(output["pred"][i].item())

            t_val = "UNK"
            p_val = "UNK"
            if t_loc < len(sample["memory_values"][i]):
                t_val = sample["memory_values"][i][t_loc]
            if p_loc < len(sample["memory_values"][i]):
                p_val = sample["memory_values"][i][p_loc]
        
            results[key] = (t_val, p_val, t_loc == p_loc)
    
    return results

#%%
def results_to_dial(dev_ds, results):
    # index..
    index = {}
    for d in dev_ds:
        index[d["dialogue_id"]] = d
    
    def query_turn(dial, turnid):
        return index[dial]["turns"][turnid]
    
    def query_serv(dial):
        return index[dial]["services"]
    
    dialogs = OrderedDict()
    
    # init dataset
    for res in results:
        dial, turn, serv, slot = res.split("/")
        # GOLD: set services..
        dialogs[dial] = dict(dialogue_id=dial, services=query_serv(dial), turns=[])
            
    # fill dataset
    turn_exists = set()
    slot_exists = set()
    
    for key, res in tqdm(results.items()):
        dial, turn, serv, slot = key.split("/")
        turn = int(turn)
        target_value, pred_value, iscorrect = res
        
        # create turn. turns with no slots need to be created explicitly.
        if (dial, turn) not in turn_exists:
            for t in range(0, turn + 1, 2):
                if (dial, t) not in turn_exists:
                    # GOLD: set utterances. script gives error otherwise
                    sys_gold_text = query_turn(dial, t-1)["utterance"]
                    usr_gold_text = query_turn(dial, t)["utterance"]
                    
                    sys = dict(speaker="SYSTEM", utterance=sys_gold_text, frames=[])
                    usr = dict(speaker="USER", utterance=usr_gold_text, frames=[])
                    
                    # create empty frames
                    for s in dialogs[dial]["services"]:
                        state = dict(active_intent="", slot_values={}, requested_slots=[])
                        frame = dict(service=s, state=state, slots=[])
                        usr["frames"].append(frame)
                        
                    dialogs[dial]["turns"].append(usr)
                    dialogs[dial]["turns"].append(sys)
                    turn_exists.add((dial, t))
            
        # fill slots
        state = dialogs[dial]["turns"][turn]
        for frame in state["frames"]:
            if frame["service"] == serv:
                if (dial, turn, serv, slot) not in slot_exists:
                    frame["state"]["slot_values"][slot] = [pred_value]
                    slot_exists.add((dial, turn, serv, slot))
    
    return list(dialogs.values())

#%%
dev_results = predictor(model, dev_ds, torch_device)

raw_dev_ds = []
for fname in sorted(glob.glob("../data/dev/dialogues*.json")):
    with open(fname) as f:
        ds_list = json.load(f)
    raw_dev_ds.extend(ds_list)

dial_results = results_to_dial(raw_dev_ds, dev_results)

with open("../out/out2-dev/dialogues.json", "w") as f:
    json.dump(dial_results, f, indent=2)

print("Saved dev results")

#%%
train_results = predictor(model, train_ds, torch_device)

raw_train_ds = []
for fname in sorted(glob.glob("../data/train/dialogues*.json")):
    with open(fname) as f:
        ds_list = json.load(f)
    raw_train_ds.extend(ds_list)


#%%
tr_dial_results = results_to_dial(raw_train_ds, train_results)

with open("../out/out2-train/dialogues.json", "w") as f:
    json.dump(tr_dial_results, f, indent=2)

print("Saved dev results")
print("done")

#%%
# eval command
#
# >>> DEV
# python -m schema_guided_dst.evaluate --dstc8_data_dir data --prediction_dir out-dev/ --eval_set dev --output_metric_file out-dev/eval.json
#
# >>> TRAIN
# python -m schema_guided_dst.evaluate --dstc8_data_dir data --prediction_dir out-train/ --eval_set train --output_metric_file out-train/eval.json

