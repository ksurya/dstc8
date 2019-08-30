#!/usr/bin/env python
# coding: utf-8

# In[1]:
# NOTE: run with IPYTHON
print("Candidate-Selection-01")

import torch.utils.data as D
import torch
import numpy as np
import json
import csv
import glob
import re
import concurrent.futures
import os

from copy import deepcopy
from functools import lru_cache
from collections import OrderedDict
from types import SimpleNamespace
from collections.abc import Iterable
from tqdm import tqdm
from pytorch_transformers import BertTokenizer
from ipdb import set_trace


# # Data Reader

# ## Utils

# In[2]:


class Tokenizer:
    
    def __init__(self, bert):
        self.bert = bert
    
    def __call__(self, text, include_sep=True):
        tokens = self.bert.tokenize(text)
        if include_sep:
            tokens.insert(0, "[CLS]")
            tokens.append("[SEP]")
        return tokens
    
    
class TokenIndexer:
    
    def __init__(self, bert):
        self.bert = bert
        
    def inv(self, *args, **kw):
        # tokens MUST be list, tensor doesn't work.
        return self.bert.convert_ids_to_tokens(*args, **kw)
        
    def __call__(self, *args, **kw):
        return self.bert.convert_tokens_to_ids(*args, **kw)


# In[3]:


def label_binarize(labels, classes):
    # labels: np.array or tensor [batch, 1]
    # classes: [..] list of classes
    # weirdly,`sklearn.preprocessing.label_binarize` returns [1] or [0]
    # instead of onehot ONLY when executing in this script!
    vectors = [np.zeros(len(classes)) for _ in labels]
    for i, label in enumerate(labels):
        for j, c in enumerate(classes):
            if c == label:
                vectors[i][j] = 1
    return np.array(vectors)
    

def label_inv_binarize(vectors, classes):
    # labels: np.array or tensor [batch, classes]
    # classes: [..] list of classes
    # follows sklearn LabelBinarizer.inverse_transform()
    # given all zeros, predicts label at index 0, instead of returning none!
    # sklearn doesn't have functional API of inverse transform
    labels = []
    for each in vectors:
        index = np.argmax(each)
        labels.append(classes[index])
    return labels


# In[4]:


def padded_array(array, value=0):
    # TODO: this does not do type checking; and wow it can be slow on strings.
    # expects array to have fixed _number_ of dimensions
    
    # resolve the shape of padded array
    shape_index = {}
    queue = [(array, 0)]
    while queue:
        subarr, dim = queue.pop(0)
        shape_index[dim] = max(shape_index.get(dim, -1), len(subarr))
        for x in subarr:
            if isinstance(x, Iterable) and not isinstance(x, str):
                queue.append((x, dim+1))
    shape = [shape_index[k] for k in range(max(shape_index) + 1)]
    
    # fill the values 
    padded = np.ones(shape) * value
    queue = [(array, [])]
    while queue:
        subarr, index = queue.pop(0)
        for j, x in enumerate(subarr):
            if isinstance(x, Iterable):
                queue.append((x, index + [j]))
            else:
                padded[tuple(index + [j])] = x
    return padded


# ## Schema Reader

# In[5]:


class Schemas(object):

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


# ## Dialogue Reader

# In[6]:


class DialogueDataset(D.Dataset):
    def __init__(self, filename, schemas, tokenizer, token_indexer):
        with open(filename) as f:
            self.ds = json.load(f)
        self.schemas = schemas
        self.tokenizer = tokenizer
        self.token_indexer = token_indexer
        self.dialogues = []
        for dial in self.ds:
            fields = self.dial_to_fields(dial)
            self.dialogues.append(fields)
        self.schemas = None
        self.tokenizer = None
        self.token_indexer = None
            
    def __getitem__(self, idx):
        return self.dialogues[idx]
    
    def __len__(self):
        return len(self.dialogues)
    
    # Turn,Service,Intents,Value
    def fd_intent_name(self, dial, fields):
        resp = dict(value=[])
        for service in dial["services"]:
            schema = self.schemas.get(service)
            resp["value"].append(schema["intent_name"])
        return resp
    
    # Turn,Service,Intents,Tokens
    def fd_intent_desc(self, dial, fields):
        resp = dict(value=[], tokens=[], ids=[], ids_pos=[], mask=[])
        for service in dial["services"]:
            desc = self.schemas.get(service)["intent_desc"]
            tokens = [self.tokenizer(d) for d in desc]
            ids = [self.token_indexer(t) for t in tokens]
            mask = [[1] * len(d) for d in tokens]
            pos = [list(range(1, len(d)+1)) for d in tokens]
            resp["value"].append(desc)
            resp["tokens"].append(tokens)
            resp["ids"].append(ids)
            resp["ids_pos"].append(pos)
            resp["mask"].append(mask)
        return resp
    
    # Turn,Service,Value
    def fd_serv_name(self, dial, fields):
        resp = dict(value=[])
        for service in dial["services"]:
            resp["value"].append(service)
        return resp
    
    # Turn,Service,Tokens
    def fd_serv_desc(self, dial, fields):
        resp = dict(value=[], tokens=[], ids=[], ids_pos=[], mask=[])
        for service in dial["services"]:
            desc = self.schemas.get(service)["desc"]
            tokens = self.tokenizer(desc)
            ids = self.token_indexer(tokens)
            resp["value"].append(desc)
            resp["tokens"].append(tokens)
            resp["ids"].append(ids)
            resp["ids_pos"].append(list(range(1, len(ids) + 1)))
            resp["mask"].append([1] * len(ids))
        return resp
    
    # Turn,Service,Slots,Value
    def fd_slot_name(self, dial, fields):
        resp = {"value": []}
        for serv in dial["services"]:
            schema = self.schemas.get(serv)
            resp["value"].append(schema["slot_name"])
        return resp
    
    # Turn,Service,Slots,Tokens
    def fd_slot_desc(self, dial, fields):
        resp = dict(value=[], tokens=[], ids=[], ids_pos=[], mask=[])
        for serv in dial["services"]:
            s_desc = self.schemas.get(serv)["slot_desc"]
            s_tokens = [self.tokenizer(d) for d in s_desc]
            s_ids = [self.token_indexer(d) for d in s_tokens]
            s_mask = [[1] * len(d) for d in s_tokens]
            resp["value"].append(s_desc)
            resp["tokens"].append(s_tokens)
            resp["ids"].append(s_ids)
            resp["ids_pos"].append([list(range(1, len(i) + 1)) for i in s_ids])
            resp["mask"].append(s_mask)
        return resp
    
    # Turn,Service,Slot,1
    def fd_slot_iscat(self, dial, fields):
        resp = dict(value=[], ids=[], mask=[])
        for serv in dial["services"]:
            schema = self.schemas.get(serv)
            resp["value"].append(schema["slot_iscat"])
            resp["ids"].append([int(i) for i in schema["slot_iscat"]])
            resp["mask"].append([1] * len(schema["slot_iscat"]))
        return resp

    # Turn,Memory,Tokens
    def fd_slot_memory(self, dial, fields):
        # TODO: iscat feature..
        # memory is a sequence of slot tagger values across all frames in a dial.. that grows per turn
        # maintain snapshot at each turn
        resp = dict(
            value=[], # unfeaturized memory
            value_desc = [],
            value_slot = [], # slots that use this value
            tokens=[], # [turn, memory, tokens]
            tokens_desc = [],
            ids=[], # [turn, memory, tokens]
            ids_desc = [], # [turn, memory, tokens]
            ids_iscat = [],  # [turn, memory, 2] 0/1 onehot
            ids_pos=[], # [turn, memory, 1] positional info for memory cells
            ids_pos_tokens = [], # [turn, memory, tokens] position info for memory values
            ids_pos_desc = [], # [turn, memory, tokens] positional info for memory desc values
            ids_memsize=[], # mem sizes [turn, 1]
            mask=[], # mask on memory [turn, mem-size, tokens]
            mask_desc = [], # mask on mem desc [turn, mem, tokens]
        )
        
        # memory is sequential, initialized by vals from schema, only has values
        memory = ["NONE", "dontcare"] # keep these at index 0, 1
        memory_slot = [["*"], ["*"]]
        memory_desc = ["NONE", "DONTCARE"]
        memory_iscat = [[1,1], [1,1]] # [not used by cat slot, used by cat slot]
        memory_index = {}
        for serv in dial["services"]:
            schema = self.schemas.get(serv)
            servdesc = schema["desc"]
            for slotname, slotdesc, iscat, values in zip(schema["slot_name"], schema["slot_desc"], schema["slot_iscat"], schema["slot_vals"]):
                for val in values:
                    if val not in memory_index:
                        memory.append(val)
                        memory_slot.append([slotname])
                        memory_desc.append(slotdesc + "[SEP]" + servdesc)
                        memory_iscat.append([int(iscat==False), int(iscat==True)]) # LOL; I just can't believe I am not using label_binarize!!
                        memory_index[val] = len(memory)
                    else:
                        idx = memory_index[val] - 1
                        if slotname not in memory_slot[idx]:
                            memory_desc[idx] = memory_desc[idx] + "[SEP]" + slotdesc + "[SEP]" + servdesc
                            memory_slot[idx].append(slotname)
                            memory_iscat[idx][int(iscat)] = 1
        
        # at each user turn create a memory snapshot..
        for turn in dial["turns"]:
            memory = deepcopy(memory)
            memory_desc = deepcopy(memory_desc)
            memory_slot = deepcopy(memory_slot)
            memory_iscat = deepcopy(memory_iscat)
            
            # pick slot tagger's ground truth ; categorical slot vals are already initialized!
            utter = turn["utterance"]
            for frame in turn["frames"]:
                schema = self.schemas.get(frame["service"])
                servdesc = schema["desc"]
                for tag in frame["slots"]:
                    slotname = tag["slot"]
                    iscat = schema["slot_iscat"][schema["slot_name"].index(slotname)]
                    slotdesc = schema["slot_desc"][schema["slot_name"].index(slotname)] # shit! O(N*2)
                    st, en = tag["start"], tag["exclusive_end"]
                    value = utter[st:en]
                    if value not in memory_index:
                        memory.append(value)
                        memory_slot.append([slotname])
                        memory_desc.append(slotdesc)
                        memory_iscat.append([int(iscat==False), int(iscat==True)])
                        memory_index[value] = len(memory)
                    else:
                        idx = memory_index[value] - 1
                        if slotname not in memory_slot[idx]:
                            memory_slot[idx].append(slotname)
                            memory_desc[idx] = memory_desc[idx] + "[SEP]" + slotdesc + "[SEP]" + servdesc
                            memory_iscat[idx][int(iscat)] = 1
                        
            if turn["speaker"] == "USER":
                resp["value"].append(memory)
                resp["value_desc"].append(memory_desc)
                resp["value_slot"].append(memory_slot)
                resp["ids_iscat"].append(memory_iscat)
                resp["ids_memsize"].append(len(memory))

        # tokenize and index the memory values
        # value: [turn, values], ids/tokens: [turn, values, tokens]
        for mem_desc_snapshot, mem_snapshot in zip(resp["value_desc"], resp["value"]):
            mem_tokens = []
            mem_tokens_desc = []
            mem_ids = []
            mem_ids_desc = []
            mem_pos = list(range(1, len(mem_snapshot) + 1))
            mem_pos_tokens = []
            mem_pos_desc = []
            mem_mask = []
            mem_mask_desc = []
            for desc, val in zip(mem_desc_snapshot, mem_snapshot):
                # featurize memory
                tokens = self.tokenizer(val)
                mem_tokens.append(tokens)
                mem_ids.append(self.token_indexer(tokens))
                mem_mask.append([1] * len(tokens))
                mem_pos_tokens.append(list(range(1, len(tokens) + 1)))
                
                # featurize memory desc
                tokens_desc = self.tokenizer(desc)
                mem_tokens_desc.append(tokens_desc)
                mem_ids_desc.append(self.token_indexer(tokens_desc))
                mem_mask_desc.append([1] * len(tokens_desc))
                mem_pos_desc.append(list(range(1, len(tokens_desc) + 1)))
                
            resp["tokens"].append(mem_tokens)
            resp["tokens_desc"].append(mem_tokens_desc)
            resp["ids"].append(mem_ids)
            resp["ids_desc"].append(mem_ids_desc)
            resp["ids_pos"].append(mem_pos)
            resp["ids_pos_tokens"].append(mem_pos_tokens)
            resp["ids_pos_desc"].append(mem_pos_desc)
            resp["mask"].append(mem_mask)
            resp["mask_desc"].append(mem_mask_desc)
        
        return resp
    
    def fd_slot_memory_loc(self, dial, fields):
        resp = dict(
            value=[],  # string value
            ids=[], # memory loc
            ids_onehot=[], # onehot of memory loc
            mask=[],
            mask_onehot=[],
            mask_none=[], # 1 -> non NONE values
            mask_none_onehot=[],
        )
        
        # query dialog memory snapshots per turn. NOTE: fd_slot_memory should exec first.
        memory = fields["slot_memory"]["value"]
        
        # init snapshot: service, slot -> memory loc, val
        memory_loc = []
        memory_val = []
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                loc = OrderedDict()
                val = OrderedDict()
                for serv in dial["services"]:
                    loc[serv] = OrderedDict()
                    val[serv] = OrderedDict()
                    for slot in self.schemas.get(serv)["slot_name"]:
                        loc[serv][slot] = None
                        val[serv][slot] = None
                memory_loc.append(loc)
                memory_val.append(val)
        
        # fill the memory locations 
        snapshot_id = 0
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                turn_memory = memory[snapshot_id]
                turn_memory_loc = memory_loc[snapshot_id]
                turn_memory_val = memory_val[snapshot_id]
                for frame in turn["frames"]:
                    service = frame["service"]
                    for slot, values in frame["state"]["slot_values"].items():
                        val = re.sub("\u2013", "-", values[0]) # dial 59_00125 turn 14
                        turn_memory_loc[service][slot] = turn_memory.index(val)
                        turn_memory_val[service][slot] = val
                    # add locations to UNKNOWN slots
                    for slot, val in turn_memory_loc[service].items():
                        if val is None:
                            turn_memory_loc[service][slot] = turn_memory.index("NONE")
                            turn_memory_val[service][slot] = "NONE"
                snapshot_id += 1
        
        # featurize
        snapshot_id = 0
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                turn_memory = memory[snapshot_id]
                turn_memory_loc = memory_loc[snapshot_id]
                turn_memory_val = memory_val[snapshot_id]
                none_loc = memory[snapshot_id].index("NONE")
                turn_fields = dict(
                    value=[], ids=[], ids_onehot=[], mask=[], mask_onehot=[],
                    mask_none=[], mask_none_onehot=[],
                )

                for serv in turn_memory_loc:
                    mem_size = len(turn_memory)
                    vals = list(turn_memory_val[serv].values())
                    ids = list(turn_memory_loc[serv].values())
                    ids_onehots = label_binarize(ids, list(range(mem_size)))
                    mask = [1] * len(ids)
                    mask_onehots = [[1] * mem_size for _ in ids]

                    mask_none = [int(v!="NONE") for v in vals]
                    mask_none_onehot = [[1] * mem_size for _ in ids]
                    for v, loc, onehot in zip(vals, ids, mask_none_onehot):
                        if v == "NONE":
                            onehot[loc] = 0
                                
                    turn_fields["value"].append(vals)
                    turn_fields["ids"].append(ids)
                    turn_fields["ids_onehot"].append(ids_onehots)
                    turn_fields["mask"].append(mask)
                    turn_fields["mask_onehot"].append(mask_onehots)
                    turn_fields["mask_none"].append(mask_none)
                    turn_fields["mask_none_onehot"].append(mask_none_onehot)

                # update turn
                for k, v in turn_fields.items():
                    resp[k].append(v)
                
                snapshot_id += 1
            
        return resp
    
    def fd_num_turns(self, dial, fields):
        return {"ids": len(dial["turns"])}
    
    def fd_num_frames(self, dial, fields):
        return {"ids": [len(t["frames"]) for t in dial["turns"]]}
    
    def fd_usr_utter(self, dial, fields):
        resp = dict(value=[], ids=[], ids_pos=[], mask=[], tokens=[])
        for turn in dial["turns"]:
            if turn["speaker"] == "USER":
                utter = turn["utterance"]
                tokens = self.tokenizer(utter)
                ids = self.token_indexer(tokens)
                resp["value"].append(utter)
                resp["ids"].append(ids)
                resp["ids_pos"].append(list(range(1, len(ids) + 1)))
                resp["tokens"].append(tokens)
                resp["mask"].append([1] * len(tokens))
        return resp
    
    def fd_sys_utter(self, dial, fields):
        resp = dict(value=[], ids=[], ids_pos=[], mask=[], tokens=[])
        for turn in dial["turns"]:
            if turn["speaker"] == "SYSTEM":
                utter = turn["utterance"]
                tokens = self.tokenizer(utter)
                ids = self.token_indexer(tokens)
                resp["value"].append(utter)
                resp["ids"].append(ids)
                resp["ids_pos"].append(list(range(1, len(ids) + 1)))
                resp["tokens"].append(tokens)
                resp["mask"].append([1] * len(tokens))
        return resp
    
    def fd_dial_id(self, dial, fields):
        return {"value": dial["dialogue_id"]}
    
    def dial_to_fields(self, dial):
        fields = {}
        ordered_funcs = [
            "fd_dial_id",
            "fd_num_turns", "fd_num_frames",
            "fd_serv_name", "fd_serv_desc", 
            "fd_slot_name", "fd_slot_desc", "fd_slot_iscat",
            "fd_slot_memory", "fd_slot_memory_loc",
            "fd_intent_name", "fd_intent_desc",
            "fd_usr_utter", "fd_sys_utter"]
        for func in ordered_funcs:
            name = func.split("fd_", maxsplit=1)[-1]
            value = getattr(self, func)(dial, fields)
            if value is not None:
                fields[name] = value
        return fields


# ## Load Data
bert_ = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = Tokenizer(bert_)
token_indexer = TokenIndexer(bert_)

train_schemas = Schemas("../data/train/schema.json")
test_schemas = Schemas("../data/dev/schema.json")

# In[8]:

print("Loading Training Data")
# load training dataset
train_dial_sets = []
train_dial_files = sorted(glob.glob("../data/train/dialogues*.json"))
num_workers = min(20, len(train_dial_files))

def worker1(filename):
    return DialogueDataset(filename, train_schemas, tokenizer, token_indexer)

with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    for ds in tqdm(executor.map(worker1, train_dial_files), total=len(train_dial_files)):
        train_dial_sets.append(ds)

train_ds = D.ConcatDataset(train_dial_sets)


# In[9]:


# load test dataset
test_dial_sets = []
test_dial_files = sorted(glob.glob("../data/dev/dialogues*.json"))
num_workers = min(20, len(train_dial_files))

def worker2(filename):
    return DialogueDataset(filename, test_schemas, tokenizer, token_indexer)

with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    for ds in tqdm(executor.map(worker2, test_dial_files), total=len(test_dial_files)):
        test_dial_sets.append(ds)

test_ds = D.ConcatDataset(test_dial_sets)


# # Training Utils

# In[10]:


import torch.nn as nn


# In[11]:


def move_to_device(obj, device):
    if type(obj) is list:
        return [move_to_device(o, device) for o in obj]
    elif type(obj) is dict:
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif type(obj) is torch.Tensor or isinstance(obj, nn.Module):
        return obj.to(device)
    return obj


# In[12]:


def dialogue_mini_batcher(dialogues):
    default_padding = 0
    batch = {}
    for dial in dialogues:
        # populate the batch
        for field, data in dial.items():
            if field not in batch:
                batch[field] = {}
            for attr, val in data.items():
                if attr == "padding":
                    batch[field][attr] = val
                else:
                    batch[field][attr] = batch[field].get(attr, [])
                    batch[field][attr].append(val)

    # padding on field attributes
    for field_name, data in batch.items():
        for attr in data:
            if attr.startswith(("ids", "mask")):
                data[attr] = padded_array(data[attr], default_padding)
                data[attr] = torch.tensor(data[attr], device="cpu") # whatif its in device0 in epoch0, then at epoch1, sent to device1

    return batch


class DialogIterator(object):
    """A simple wrapper on DataLoader"""
    
    def __init__(self, dataset, batch_size, *args, **kw):
        self.length = None
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterator = D.DataLoader(dataset, batch_size, *args, **kw)
        
    def __len__(self):
        if self.length is None:
            self.length = 0
            num_turns = -float("inf")
            num_servs = -float("inf")
            num_slots = -float("inf")
            for i, dial in enumerate(self.dataset):
                num_turns = max(num_turns, dial["num_turns"]["ids"] // 2)
                num_servs = max(num_servs, len(dial["serv_name"]["value"]))
                num_slots = max(num_slots, sum([len(x) for x in dial["slot_name"]["value"]]))
                if i % self.batch_size == 0:
                    self.length += (num_turns * num_servs * num_slots)
                    num_turns = -float("inf")
                    num_servs = -float("inf")
                    num_slots = -float("inf")
        return self.length
    
    def __iter__(self):
        for batch in self.iterator:
            num_turns = batch["usr_utter"]["ids"].shape[1]
            num_services = batch["serv_desc"]["ids"].shape[1]
            num_slots = batch["slot_desc"]["ids"].shape[2]
            for turnid in range(num_turns):
                for serviceid in range(num_services):
                    for slotid in range(num_slots):
                        inputs = dict(turnid=turnid, serviceid=serviceid, slotid=slotid)
                        inputs.update(batch)
                        yield inputs


# In[13]:


def get_sample(ds, size=1):
    return next(iter(DialogIterator(ds, size, collate_fn=dialogue_mini_batcher)))

def print_shapes(ds):
    it = get_sample(ds, size=1)
    for field, val in it.items():
        if type(val) is dict:
            for attr in val:
                if attr.startswith("ids") or attr.startswith("mask"):
                    print(field, attr, "-->", val[attr].shape)
                    

# TODO: why mem ids, and mem loc size is diff by 1?
print_shapes(train_ds)


# # Model

# In[14]:


from pytorch_transformers import BertModel, BertConfig
from allennlp.training.metrics import BooleanAccuracy, CategoricalAccuracy
from collections import OrderedDict

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch


# In[52]:


class BertDownstream(nn.Module):
    # https://huggingface.co/pytorch-transformers/model_doc/bert.html
    # https://github.com/hanxiao/bert-as-service/blob/master/docs/section/faq.rst#id6
    
    def __init__(self, btype, requires_grad=False):
        super().__init__()
        self.emb = BertModel.from_pretrained(btype, output_hidden_states=True)
        for name, param in self.emb.named_parameters():
            param.requires_grad = requires_grad

    @property
    def output_dim(self):
        return 768
    
    def forward(self, input_ids, position_ids=None, type_ids=None, attention_mask=None, flat=(1,-1)):
        sh = list(input_ids.shape)
        
        input_ids = input_ids.flatten(*flat).long()
        if position_ids is not None:
            position_ids = position_ids.flatten(*flat).long()
        if attention_mask is not None:
            attention_mask = attention_mask.flatten(*flat).long()
        if type_ids is not None:
            type_ids = type_ids.flatten(*flat).long()
        outputs = self.emb(input_ids, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=type_ids)
        
        # average of hidden layers: bs->bse
        pooled = torch.mean(torch.stack([outputs[2][i] for i in range(2, 11)]), dim=0)
        
        # unflatten
        pooled = pooled.view(sh + [-1])
        
        return pooled # (..., e)


class CandidateSelector(nn.Module):
        
    def __init__(self):
        super().__init__()
        self.emb = BertDownstream("bert-base-uncased")
        emb_dim = self.emb.output_dim
        
        self.l1 = nn.GRU(emb_dim, emb_dim, batch_first=True) # encode memory
        self.l2 = nn.GRU(emb_dim, emb_dim, batch_first=True) # encode query
        
        self.enc_pos_trn = nn.Linear(1, emb_dim)
        self.enc_pos_mem = nn.Linear(1, emb_dim)
        
        # attentions
        self.l4 = nn.Linear(emb_dim, 1)
         
        self.acc = CategoricalAccuracy()
        
        self.register_buffer("state_query", None)
        self.register_buffer("state_memory", None)
        
        # init weights: classifier's performance changes heavily on these
        for name, param in self.named_parameters():
            if name.startswith(("l1.", "l2.")):
                print("Initializing bias/weights of ", name)
                param.data.normal_(0, 0.02)
                #nn.init.xavier_uniform_(param)

    def get_metrics(self, reset=False):
        return dict(
            acc=self.acc.get_metric(reset), # avg per slot
        )
    
    def encode_query(self, batch):
        turnid = batch["turnid"]
        servid = batch["serviceid"]
        slotid = batch["slotid"]
        
        # create a query
        usr_utter = batch["usr_utter"]["ids"][:,turnid,...] # bs
        sys_utter = batch["sys_utter"]["ids"][:,turnid,...] if turnid > 0 else torch.zeros(usr_utter.shape, device=usr_utter.device, dtype=usr_utter.dtype)
        slot_desc = batch["slot_desc"]["ids"][:,servid,slotid,...] # bs
        serv_desc = batch["serv_desc"]["ids"][:,servid,...] # bs
        slot_iscat = batch["slot_iscat"]["ids"][:,servid,slotid,None].repeat(1,slot_desc.shape[-1]).type(usr_utter.dtype) # be
        
        usr_pos = batch["usr_utter"]["ids_pos"][:,turnid,...] # bs
        sys_pos = batch["sys_utter"]["ids_pos"][:,turnid,...] if turnid > 0 else torch.zeros(usr_utter.shape, device=usr_utter.device, dtype=usr_utter.dtype)
        slot_pos = batch["slot_desc"]["ids_pos"][:,servid,slotid,...] # bs
        serv_pos = batch["serv_desc"]["ids_pos"][:,servid,...] # bs
        iscat_pos = torch.ones(slot_iscat.shape, device=slot_iscat.device, dtype=slot_iscat.dtype) # be
        
        usr_mask = batch["usr_utter"]["mask"][:,turnid,...] # bs
        sys_mask = batch["sys_utter"]["mask"][:,turnid,...] if turnid > 0 else torch.zeros(usr_utter.shape, device=usr_utter.device, dtype=usr_utter.dtype)
        slot_mask = batch["slot_desc"]["mask"][:,servid,slotid,...] # bs
        serv_mask = batch["serv_desc"]["mask"][:,servid,...] # bs
        iscat_mask = batch["slot_iscat"]["mask"][:,servid,slotid,None].repeat(1,slot_desc.shape[-1]).type(usr_utter.dtype) # bs
        
        usr_type = torch.zeros(usr_utter.shape, device=usr_utter.device, dtype=usr_utter.dtype) # bs
        sys_type = torch.zeros(sys_utter.shape, device=usr_utter.device, dtype=usr_utter.dtype).fill_(1)
        slot_type = torch.zeros(slot_desc.shape, device=usr_utter.device, dtype=usr_utter.dtype).fill_(2)
        serv_type = torch.zeros(serv_desc.shape, device=usr_utter.device, dtype=usr_utter.dtype).fill_(3)
        iscat_type = torch.zeros(slot_iscat.shape, device=usr_utter.device, dtype=usr_utter.dtype).fill_(4)
        
        query = torch.cat([usr_utter, sys_utter, serv_desc, slot_iscat, slot_desc], dim=1) # b4s
        query_pos = torch.cat([usr_pos, sys_pos, serv_pos, iscat_pos, slot_pos], dim=1) # b4s
        query_mask = torch.cat([usr_mask, sys_mask, serv_mask, iscat_mask, slot_mask], dim=1) # b4s
        query_type = torch.cat([usr_type, sys_type, serv_type, iscat_type, slot_type], dim=1) # b4s
        
        query_emb = self.emb(query, position_ids=query_pos, attention_mask=query_mask) # b4se
        
        query_o, query_h = self.l1(query_emb)
        #self.state_query = query_h.detach()
        
        return query_h[-1] # be
        
    def encode_memory(self, batch):
        turnid = batch["turnid"]
        serviceid = batch["serviceid"]
        slotid = batch["slotid"]
        
        # Encode memory
        mem = batch["slot_memory"]["ids"][:,turnid,:] # bms
        mem_mask = batch["slot_memory"]["mask"][:,turnid,:] 
        mem_pos = batch["slot_memory"]["ids_pos_tokens"][:,turnid,:] # position of memory cell, not tokens. bm
        sh = mem.shape
        
        mem = self.emb(mem, position_ids=mem_pos, attention_mask=mem_mask, flat=(0,1)) # bmse

        mem, mem_h = self.l2(mem.flatten(0,1))
        #self.state_memory = mem_h.detach()
        return mem_h[-1].view(sh[0], sh[1], -1) # bme
    
    def get_value(self, memory, query, batch, layers=2):
        # encode dialog pos -- time
        turnid = batch["turnid"]
        turn_pos = torch.zeros(query.shape[0], 1, device=query.device).fill_(turnid).float()
        turn_pos = self.enc_pos_trn(turn_pos) # be
        
        # encode memory pos -- time
        mem_pos = batch["slot_memory"]["ids_pos"][:,turnid,:].float() # bm
        sh = mem_pos.shape
        mem_pos = F.relu(self.enc_pos_mem(mem_pos.unsqueeze(-1)))
        
        query = query + turn_pos
        memory = memory + mem_pos
        
        for _ in range(layers):
            attn = F.softmax(torch.bmm(memory, query.unsqueeze(-1)), 1)
            output = (memory * attn).sum(1)
            query = query + output
            
        # memory: bme, query: be
        output = torch.einsum("bme,be->bme", memory, query)
        output = self.l4(output).squeeze(-1)
        output = F.softmax(output, -1)
        
        return output, attn # bm
    
    
    def calc_acc(self, preds, targets, mask):
        # preds: b,mem, targets: b, mask: m
        pred_ids = torch.argmax(preds, -1).float()
        match = (pred_ids * mask) == (targets * mask)
        val = (match.float() * mask).sum()
        total = mask.sum()
        if total == 0:
            val.fill_(0)
        else:
            val /= total
        return val
        
    def forward(self, **batch):
        turnid = batch["turnid"]
        serviceid = batch["serviceid"]
        slotid = batch["slotid"]
        
        # doc: GRU outputs [batch, seq, emb * dir] [layers * dir, batch, emb]

        # get fixed size encodings
        query = self.encode_query(batch)
        memory = self.encode_memory(batch)
        
        score, attn = self.get_value(memory, query, batch)
        
        output = {"score": score, "mem_attn": attn} # bm

        if "slot_memory_loc" in batch:
            target = batch["slot_memory_loc"]
            
            # calc loss
            target_score_oh = target["ids_onehot"][:,turnid,serviceid,slotid,:].float() # bm
            target_mask_oh = target["mask_onehot"][:,turnid,serviceid,slotid,:].float()
            output["loss"] = F.binary_cross_entropy(score, target_score_oh, target_mask_oh).unsqueeze(0) # don't return scalar
            
            # log metrics
            target_ids = target["ids"][:,turnid,serviceid,slotid].float() # b
            target_mask = (target["mask"][:,turnid,serviceid,slotid]).float()
            self.acc(score, target_ids, target_mask)
            
            # output tensors
            output["target_ids"] = target_ids
            output["pred_ids"] = torch.argmax(score, dim=-1)
            output["mask"] = target["mask"][:,turnid,serviceid,slotid].float()
            output["mask_none"] = target["mask_none"][:,turnid,serviceid,slotid].float()
            output["acc"] = self.calc_acc(score, target_ids, target_mask).unsqueeze(0)
            
        return output


# # Trainer

# In[16]:


from allennlp.training.tensorboard_writer import TensorboardWriter
import time


# In[17]:


from tabulate import tabulate

def print_results(results):
    header = list(results[0].keys())
    values = [list(x.values()) for x in results]
    print(tabulate(values, header, "fancy_grid"))
    

def get_predicted_labels(b_input, b_output):
    num_batches = b_input["usr_utter"]["ids"].shape[0]
    turnid = b_input["turnid"]
    servid = b_input["serviceid"]
    slotid = b_input["slotid"]
    
    batch_results = []
    
    for b in range(num_batches):
        try:
            target_loc = int(b_output["target_ids"][b].item())
            target_val1 = b_input["slot_memory_loc"]["value"][b][turnid][servid][slotid]
            target_val2 = b_input["slot_memory"]["value"][b][turnid][target_loc]
            assert target_val1 == target_val2

            pred_loc = int(b_output["pred_ids"][b].item())
            pred_val = b_input["slot_memory"]["value"][b][turnid][pred_loc]
        
            item = OrderedDict(
                dial_id=b_input["dial_id"]["value"][b],
                turnid=turnid,
                serv_name=b_input["serv_name"]["value"][b][servid],
                slot_name=b_input["slot_name"]["value"][b][servid][slotid],
                slot_loc=target_loc,
                slot_val=target_val1,
                pred_slot_loc=pred_loc,
                pred_slot_val=pred_val,
                correct=pred_loc==target_loc,
            )
            batch_results.append(item)
        except:
            print("Unbounded results")
        
    return batch_results

# iterate over the dialog..
# results = get_predicted_labels(sample, output)
# print_results(results)


# In[42]:


def module(m):
    if type(m) is nn.DataParallel:
        return m.module
    return m

def train(model, optimizer, batch_size, num_epochs, train_ds, test_ds, device):
    model = model.train()
    current_iter = 0
    tensorboard = TensorboardWriter(
        get_batch_num_total=lambda: current_iter,
        summary_interval=10,
        serialization_dir="../data/tensorboard/exp0"
    )
    
    histogram_weights = [name for name, p in model.named_parameters() if not ".emb." in name]
    
    for epoch in range(num_epochs):
        train_iter = DialogIterator(train_ds, batch_size, collate_fn=dialogue_mini_batcher)
        num_batches = len(train_iter) 
        if test_ds:
            test_iter = DialogIterator(test_ds, batch_size, collate_fn=dialogue_mini_batcher)
            num_batches += len(test_iter)
        
        with tqdm(total=num_batches) as pbar:
            # train
            pbar.set_description("Train {}".format(epoch))
            model = model.train()
            train_iter = DialogIterator(train_ds, batch_size, collate_fn=dialogue_mini_batcher)
            
            # to know when the dialog and service changes
            prev_turnid = -1
            
            # avg across turn
            turn_slot_acc = 0
            turn_loss = 0
            num_turns = 0
            metrics = OrderedDict()
            
            for i, batch in enumerate(train_iter):
                batch = move_to_device(batch, device)
                current_iter += 1
                
                optimizer.zero_grad()
                output = model(**batch)
                output["loss"] = output["loss"].mean()
                
                if np.random.normal(0, 1, size=1) > 0 and (output["mask"] * output["mask_none"]).sum() == 0:
                    # downsampled the batch..
                    prev_turnid = batch["turnid"]
                    pbar.update(1)
                    continue
                
                output["loss"].backward()
                optimizer.step()

                # update acc/loss counters
                if (output["mask"]).sum() > 0:
                    turn_slot_acc += output["acc"].mean().item()
                    turn_loss += output["loss"].item()
                    num_turns += 1

                # DEBUG point
                if output["acc"] == 1 and ((output["mask"] * output["mask_none"]) == 1).all():
                    res = get_predicted_labels(batch, output)
                    print_results(res)

                # At every new turn
                if batch["serviceid"] == 0 and batch["slotid"] == 0 and batch["turnid"] != prev_turnid:
                    # update progress bar
                    metrics["loss"] = turn_loss / num_turns
                    metrics["slot_acc"] = turn_slot_acc / num_turns
                    num_turns = 0
                    turn_loss = 0
                    turn_slot_acc = 0

                    # update tensorboard logs
                    tensorboard.add_train_scalar("loss", metrics["loss"], timestep=current_iter)
                    tensorboard.log_metrics(train_metrics=metrics, epoch=current_iter)
                    tensorboard.add_train_histogram("target_ids", output["target_ids"])
                    tensorboard.add_train_histogram("pred_ids", output["pred_ids"])
                    tensorboard.add_train_histogram("mem_attn", output["mem_attn"])
                    tensorboard.log_parameter_and_gradient_statistics(model, None)

                metrics.update(turn=batch["turnid"], serv=batch["serviceid"], slot=batch["slotid"])
                pbar.set_postfix(metrics)
                    
                prev_turnid = batch["turnid"]
                pbar.update(1)

            # test
            if test_ds:
                pbar.set_description("Test {}".format(epoch))
                with torch.no_grad():
                    model = model.eval()
                    prev_turnid = -1
                    for i, batch in enumerate(test_iter):
                        metrics = OrderedDict()
                        batch = move_to_device(batch, device)
                        current_batch += 1
                        output = model(**batch)
                        output["loss"] = output["loss"].mean()
                        # at each new turn
                        if prev_turnid != batch["turnid"]:
                            metrics.update(module(model).get_metrics(reset=True, goal_reset=True))
                        else:
                            curr_met = module(model).get_metrics(reset=True)
                            metrics.update(acc=curr_met["acc"])
                        metrics["loss"] = output["loss"].item()
                        metrics.update(turnid=batch["turnid"], servid=batch["serviceid"], slotid=batch["slotid"])
                        pbar.set_postfix(metrics)
                        prev_turnid = batch["turnid"]
                        pbar.update(1)


# In[49]:


print("Remove tensorboard logs at exp0/")
get_ipython().system('rm -rf ../data/tensorboard/exp0/*')

print("set number of devices") # not sure we can in jupyter once program already kicked in the first time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = "cuda"

print("loading model")
model = CandidateSelector()
#model = nn.DataParallel(model)
model = move_to_device(model, device)

optim = torch.optim.Adam(model.parameters(), lr=3e-5)
train_samples = train_ds
test_samples = [test_ds[i] for i in range(10)]

print("started training")
train(
    model=model,
    optimizer=optim,
    train_ds=train_samples,
    test_ds=None, #test_samples,
    device=device,
    num_epochs=2,
    batch_size=32,
)

print("Saving model...")
torch.save(model, "../data/candidate-selection-01.pkl")
