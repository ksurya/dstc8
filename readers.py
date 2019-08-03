import glob
import os
import json
import numpy as np

from functools import lru_cache
from sklearn.preprocessing import label_binarize

from allennlp.data import Instance
from allennlp.data.fields import TextField, ListField, MetadataField, ArrayField, MultiLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import PretrainedBertIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter


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


class DialogueDatasetReader(DatasetReader):

    def __init__(self, limit=float("inf"), lazy=False):
        super().__init__(lazy)
        self.files_limit = limit
        self.token_indexers = {"tokens": PretrainedBertIndexer("bert-base-uncased")}
        self.tokenizer = BertBasicWordSplitter()
        self.schema = None

    def _read(self, files_path):
        self.schemas = SchemaList(os.path.join(files_path, "schema.json"))
        for count, filename in enumerate(glob.glob(os.path.join(files_path, "dialogues*.json"))):
            if count < self.files_limit:
                with open(filename) as f:
                    dialogs = json.load(f)
                    for d in dialogs:
                        yield self.text_to_instance(d)

    def to_text_field(self, sent):
        tokens = self.tokenizer.split_words(sent)
        return TextField(tokens, self.token_indexers)

    def to_speaker_field(self, turnid, dialogue, info):
        return self.to_text_field(dialogue["turns"][turnid]["speaker"])
    
    def to_utter_field(self, turnid, dialogue, info):
        return self.to_text_field(dialogue["turns"][turnid]["utterance"])
    
    def to_sys_utter_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "SYSTEM":
            return self.to_text_field(dialogue["turns"][turnid]["utterance"])
        
    def to_usr_utter_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            return self.to_text_field(dialogue["turns"][turnid]["utterance"])

    def to_service_name_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            return MetadataField([
                x["service"] for x in turn["frames"]
            ])

    def to_service_desc_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            service_desc = []
            for frame in turn["frames"]:
                d =  self.schemas.get_service_desc(frame["service"])
                service_desc.append(self.to_text_field(d))
            return ListField(service_desc)
        
    def to_intent_name_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            return MetadataField([
                x["state"]["active_intent"] for x in turn["frames"]
            ])
    
    def to_intent_desc_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            intent_desc = []
            for frame in turn["frames"]:
                service = frame["service"]
                intent = frame["state"]["active_intent"]
                d = self.schemas.get_intent_desc(service, intent)
                intent_desc.append(self.to_text_field(d or "NONE"))
            return ListField(intent_desc)
    
    def to_intent_exist_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            intent_exist = [] # across service
            for frame in turn["frames"]:
                service = frame["service"]
                intent = frame["state"]["active_intent"]
                encoding = label_binarize([intent], classes=info["each"][service]["intent_name"])
                intent_exist.append(ArrayField(encoding))
            return ListField(intent_exist)
    
    def to_all_intent_exist_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            intent_exist = [] # across service
            for frame in turn["frames"]:
                intent = frame["state"]["active_intent"]
                encoding = label_binarize([intent], classes=info["all"]["intent_name"])
                intent_exist.append(encoding)
            intent_exist = np.concatenate(intent_exist)
            return ArrayField(intent_exist)

    def to_all_service_exist_field(self, turnid, dialogue, info):
        turn = dialogue["turns"][turnid]
        if turn["speaker"] == "USER":
            all_service_exist = []
            for frame in turn["frames"]:
                encoding = label_binarize([frame["service"]], classes=info["all"]["service_name"])
                all_service_exist.append(encoding)
            all_service_exist = np.concatenate(all_service_exist)
            return ArrayField(all_service_exist)

    def text_to_instance(self, dialogue):
        # schema info
        info = {
            "each": {}, # by service name
            "all": {},  # services merged by fields
        }
        for service in dialogue["services"]:
            info["each"][service] = self.schemas.get(service)
            for k, v in info["each"][service].items():
                info["all"][k] = info["all"].get(k, [])
                info["all"][k].extend(v)

        # add turn level fields
        fields = dict(
            speaker=[],
            utter=[],
            sys_utter=[],
            usr_utter=[],

            intent_name=[],
            intent_desc=[],
            intent_exist=[],  # one hot encoding across service

            # These two are probably not useful, but leaving here..
            service_name=[],
            service_desc=[],

            all_intent_exist=[], # one hot encoding across all services in dialog
            all_service_exist=[], # one hot encoding across all services in dialog
        )
        
        # get turn level info
        for turnid in range(len(dialogue["turns"])):
            for name in fields:
                value = getattr(self, f"to_{name}_field")(turnid, dialogue, info)
                if value is not None:
                    fields[name].append(value)
        
        # add dialogue level fields
        fields["all_service_name"] = [MetadataField(i) for i in info["all"]["service_name"]]
        fields["all_service_desc"] = [self.to_text_field(i) for i in info["all"]["service_desc"]]
        fields["all_intent_name"] = [MetadataField(i) for i in info["all"]["intent_name"]]
        fields["all_intent_desc"] = [self.to_text_field(i) for i in info["all"]["intent_desc"]]
        
        # fields..
        for f in fields:
            fields[f] = ListField(fields[f])
            
        return Instance(fields)
