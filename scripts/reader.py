import glob
import os
import json
import numpy as np

from collections import OrderedDict
from functools import lru_cache

from allennlp.data import Instance, Token
from allennlp.data.fields import TextField, ListField, MetadataField, ArrayField, IndexField, Field
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter


def label_binarize(labels, classes):
    # labels: np.array or tensor [batch, classes]
    # classes: [..] list of classes
    # weirdly,`sklearn.preprocessing.label_binarize` returns [1] or [0]
    # instead of onehot ONLY when executing in this script!
    vectors = [np.zeros(len(classes)) for _ in classes]
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


class SchemaList(object):

    def __init__(self, filepath):
        with open(filepath) as f:
            self.index = {}
            for schema in json.load(f):
                service_name = schema["service_name"]
                self.index[service_name] = schema

    def get_service_desc(self, service):
        return self.index[service]["description"]

    @lru_cache(maxsize=None)
    def get_slot_desc(self, service, slot):
        for item in self.index[service]["slots"]:
            if item["name"] == slot:
                return item["description"]

    @lru_cache(maxsize=None)
    def get_intent_desc(self, service, intent):
        for item in self.index[service]["intents"]:
            if item["name"] == intent:
                return item["description"]

    @lru_cache(maxsize=None)
    def get(self, service):
        result = dict(
            # service
            service_name=service,
            service_desc=self.index[service],
            
            # slots
            slot_name=[],
            slot_desc=[],
            slot_iscat=[], 
            slot_catvals=[], # collected only for cat slots.. not sure if that makes sense

            # intents
            intent_name=[],
            intent_desc=[],
            intent_iscat=[],
            intent_istrans=[],
            intent_reqslots=[],
            intent_optslots=[],
            intent_optvals=[],
        )

        for slot in self.index[service]["slots"]:
            result["slot_name"].append(slot["name"])
            result["slot_desc"].append(slot["description"])
            result["slot_iscat"].append(slot["is_categorical"])
            result["slot_catvals"].append(slot["possible_values"] if slot["is_categorical"] else [])
        
        for intent in self.index[service]["intents"]:
            result["intent_name"].append(intent["name"])
            result["intent_desc"].append(intent["description"])
            result["intent_istrans"].append(intent["is_transactional"])
            result["intent_reqslots"].append(intent["required_slots"])
            result["intent_optslots"].append(list(intent["optional_slots"].keys()))
            result["intent_optvals"].append(list(intent["optional_slots"].values()))

        return result    


class DialogueReader(DatasetReader):
    """Each instance encodes a full dialogue, allowing the trainer
    to process turns and services sequentially. Turns could be merged
    into batch to process otherwise."""

    def __init__(self, limit=float("inf"), lazy=False):
        super().__init__(lazy)
        self.limit = limit
        self.token_indexers = {"tokens": PretrainedBertIndexer("bert-base-uncased")}
        self.tokenizer = BertBasicWordSplitter()
        self.schemas = None

    def _read(self, file_path):
        dirpath = os.path.split(file_path)[0]
        self.schemas = SchemaList(os.path.join(dirpath, "schema.json"))
        with open(file_path) as f:
            dialogs = json.load(f)
            for d in dialogs:
                yield self.text_to_instance(d)

    def text_field(self, sent):
        tokens = self.tokenizer.split_words(sent)
        return TextField(tokens, self.token_indexers)

    def field_dialogue_id(self, dialogue):
        return MetadataField(dialogue["dialogue_id"])

    def field_turn_speaker(self, turnid, dialogue):
        return self.text_field(dialogue["turns"][turnid]["speaker"])
    
    def field_turn_utter(self, turnid, dialogue):
        return self.text_field(dialogue["turns"][turnid]["utterance"])
    
    def field_turn_sys_utter(self, turnid, dialogue):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "SYSTEM":
            return self.text_field(dialogue["turns"][turnid]["utterance"])
        
    def field_turn_usr_utter(self, turnid, dialogue):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            return self.text_field(dialogue["turns"][turnid]["utterance"])

    def field_service(self, dialogue):
        return MetadataField(dialogue["services"])

    def field_service_desc(self, dialogue):
        desc_list = []
        for service in dialogue["services"]:
            desc =  self.schemas.get_service_desc(service)
            desc_list.append(self.text_field(desc))
        return ListField(desc_list)

    def field_turn_service_exist(self, turnid, dialogue):
        turn = dialogue["turns"][turnid]
        services = dialogue["services"]
        # order frames by dialog.services list, to establish one to one mappings across fields
        frames = sorted(turn["frames"], key=lambda x: services.index(x["service"]))
        exists_onehot = label_binarize([f["service"] for f in frames], classes=services)
        exists = np.sum(exists_onehot, axis=0) # eg: [1, 0, 1, 0]
        return ArrayField(exists, padding_value=-1)
        
    def field_intent(self, dialogue):
        # NOTE: we might want to copy these lists, because if they are changed in an epoch,
        # the next epoch will not retain the changes
        intent_list = [self.schemas.get(s)["intent_name"] + ["NoIntent"] for s in dialogue["services"]]
        return MetadataField(intent_list)

    def field_intent_desc(self, dialogue):
        desc_list = []
        for service in dialogue["services"]:
            desc = [self.text_field(d) for d in self.schemas.get(service)["intent_desc"]]
            desc.append(self.text_field("NONE")) # for NoIntent 
            desc_list.append(ListField(desc))
        return ListField(desc_list)

    def field_turn_intent_exist(self, turnid, dialogue):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            # maintain order of services: service -> onehot
            exists_onehot = OrderedDict()
            for service in dialogue["services"]:
                exists_onehot[service] = None
            
            # fill encodings of existing services
            # this _will_ be onehot assuming each service has only one intent!
            for frame in turn["frames"]:
                service = frame["service"]
                all_intents = self.schemas.get(service)["intent_name"] + ["NoIntent"]
                intent = frame["state"]["active_intent"]
                encoding = label_binarize([intent], classes=all_intents)[0]
                exists_onehot[service] = ArrayField(encoding, padding_value=-1)
            
            # fill with empty encodings for remaining
            for service in exists_onehot:
                if exists_onehot[service] is None:
                    all_intents = self.schemas.get(service)["intent_name"] + ["NoIntent"]
                    encoding = np.array([0] * len(all_intents))
                    encoding[-1] = 1 # set 1 to NoIntent
                    exists_onehot[service] = ArrayField(encoding, padding_value=-1)

            return ListField(list(exists_onehot.values()))

    def field_turn_intent_changed(self, turnid, dialogue):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            # assumes system turn is always followed by user turn
            prev_user_turn = dialogue["turns"][turnid-2] if turnid >= 2 else turn
            # maintain order of services: service -> changed
            intent_changed = OrderedDict()
            for service in dialogue["services"]:
                intent_changed[service] = 0
            
            for frame, prevframe in zip(turn["frames"], prev_user_turn["frames"]):
                service = frame["service"]
                intent = frame["state"]["active_intent"]
                prev_intent = prevframe["state"]["active_intent"]
                intent_changed[service] = int(intent == prev_intent)
            
            values = list(intent_changed.values())
            return ArrayField(np.array(values), padding_value=-1)

    def field_slots(self, dialogue):
        slot_list = []
        for service in dialogue["services"]:
            slots = self.schemas.get(service)["slot_name"]
            slot_list.append(slots)
        return MetadataField(slot_list)

    def field_slots_desc(self, dialogue):
        desc_list = []
        for service in dialogue["services"]:
            desc = [self.text_field(d) for d in self.schemas.get(service)["slot_desc"]]
            desc_list.append(ListField(desc))
        return ListField(desc_list)

    def field_slots_iscat(self, dialogue):
        # TODO: use IndexField here?
        iscat_list = []
        for service in dialogue["services"]:
            iscat = np.array([int(i) for i in self.schemas.get(service)["slot_iscat"]])
            iscat_list.append(ArrayField(iscat, padding_value=-1))
        # use ListField because each element may have different shape 
        return ListField(iscat_list)

    def field_num_turns(self, dialogue):
        return MetadataField(len(dialogue["turns"]))

    def field_turn_num_frames(self, turnid, dialogue):
        return MetadataField(len(dialogue["turns"][turnid]["frames"]))

    def text_to_instance(self, dialogue):
        # initialize turn fields with list, dialog level fields with None
        fields = dict(
            dialogue_id=None, # [Batch,]
            num_turns=None, # [Batch,]
            num_frames=[], # [Batch, Turn] equals to number of services per turn

            # messages
            speaker=[], # [Batch, Turn]
            utter=[], # [Batch, Turn, Tokens]
            sys_utter=[], # [Batch, Turn, Tokens] only system utters
            usr_utter=[], # [Batch, Turn, Tokens] only user utters

            # services
            service=None, # [Batch, Service] all dialog services
            service_desc=None, # [Batch, Service, Tokens] service descriptions
            service_exist=[], # [Batch, Turn, Service] binarized
            
            # intents
            intent=None, # [Batch, Service, Intent]
            intent_desc=None, # [Batch, Service, Intent, Tokens]
            intent_exist=[], # [Batch, Turn, Service, Intent]
            intent_changed=[], # [Batch, Turn, Service]

            # state slots
            slots=None, # [Batch, Service, Slot]
            slots_desc=None, # [Batch, Service, Slot, Tokens]
            slots_iscat=None, # [Batch, Service, Slot]
        )

        # fill turn level fields
        for turnid in range(len(dialogue["turns"])):
            for name in fields:
                if type(fields[name]) is list:
                    value = getattr(self, f"field_turn_{name}")(turnid, dialogue)
                    if value is not None:
                        fields[name].append(value)
        
        for name in fields:
            if type(fields[name]) is list:
                fields[name] = ListField(fields[name])

        # fill dialog level fields
        for name in fields:
            if fields[name] is None:
                value = getattr(self, f"field_{name}")(dialogue)
                if value is not None:
                    fields[name] = value
        
        return Instance(fields)
