import os
import time
import stanza
import pdb
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from flair.data import Sentence
from flair.models import SequenceTagger
import boto3
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from collections import defaultdict
import torch

# you need to set the azure key and endpoint by yourself
key = "YourKey"
endpoint = "https://nerazure.cognitiveservices.azure.com/"

class APIS:
    def __authenticate_client(self):
        ta_credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(
                endpoint=endpoint, 
                credential=ta_credential)
        return text_analytics_client

    def azure(self, sentence):
        # create a word list (splited by whitespaces)
        client = self.__authenticate_client()
        word_list = sentence.split(" ")
        
        connected = False
        while not connected:
            print("waiting !!")
            time.sleep(0.01)
            try:
                connected = True
                sentence_list = []
                sentence_list.append(sentence)
                result = client.recognize_entities(documents = sentence_list)[0]

            except Exception as err:
                connected = False
                print("Encountered exception. {}".format(err))
        
        temp = {}
        entity_pos_info = {}
        for entity in result.entities:
            entity_pos_info[int(entity.offset)] = entity
            temp[entity.text] = entity.category
        # map the words in the word_list to the character offset, map word index to string offset (begining of the word)
        charOffset2WordID = {}
        current_offset = 0
        for index, word in enumerate(word_list):
            # add up the offset if whitespace in the string
            while current_offset<len(sentence) and sentence[current_offset] == " ":
                current_offset += 1
            charOffset2WordID[current_offset] = index
            current_offset += len(word)
        # create entity_info, which records the information of the entity and their locations
        entity_list = []
        for k, v in entity_pos_info.items():
            entity_name = v.text
            entity_type = v.category
            # if we cannot find the key in the mapping of charater id and word id, just omit this entity
            if k in charOffset2WordID:
                entity_id_start = charOffset2WordID[k]
                name_length = len(entity_name.split(" "))
                entity_id_list = []
                for i in range(0, name_length):
                    entity_id_list.append(entity_id_start+i)
                entity_list.append((entity_name, entity_type, entity_id_list))
        result = {}
        result["sentence"] = word_list
        result["entity"] = entity_list
        return result
    
    # extract entity for the AWS API
    def aws(self, sentence, comprehend_client = boto3.client('comprehend')):
        # create a word list (splited by whitespaces)
        # pdb.set_trace()
        word_list = sentence.split(" ")
        
        # collect entity info
        entity_pos_info = {}
        entities = comprehend_client.detect_entities(Text = sentence, LanguageCode = 'en')
        for entity in entities["Entities"]:
            entity_pos_info[entity["BeginOffset"]] = entity
        
        # map the words in the word_list to the character offset, map word index to string offset (begining of the word)
        charOffset2WordID = {}
        current_offset = 0
        for index, word in enumerate(word_list):
            # add up the offset if whitespace in the string
            while current_offset<len(sentence) and sentence[current_offset] == " ":
                current_offset += 1
            charOffset2WordID[current_offset] = index
            current_offset += len(word)
        # create entity_info, which records the information of the entity and their locations
        entity_list = []
        for k, v in entity_pos_info.items():
            entity_name = v["Text"]
            entity_type = v["Type"]
            # if we cannot find the key in the mapping of charater id and word id, just omit this entity
            if k in charOffset2WordID:
                entity_id_start = charOffset2WordID[k]
                name_length = len(entity_name.split(" "))
                entity_id_list = []
                for i in range(0, name_length):
                    entity_id_list.append(entity_id_start+i)
                entity_list.append((entity_name, entity_type, entity_id_list))
        result = {}
        result["sentence"] = word_list
        result["entity"] = entity_list
        return result
    
    def flair(self, sentence, tagger): # "flair/ner-english-large
    # def flair(self, sentence, tagger_name): # "flair/ner-english-large
        # tagger = SequenceTagger.load(tagger_name)
        # pdb.set_trace()
        sentence = Sentence(sentence)
        tagger.predict(sentence)
        # get word list and entity info
        word_list = []
        entity_list = []
        for token in sentence:
            word_list.append(token.text)
        for entity in sentence.get_spans('ner'):
            temp_list = []
            # entity text
            temp_list.append(entity.text)
            # entity type
            temp_list.append(entity.get_label("ner").value)
            # entity indexes
            temp_temp_list = []
            for token in entity:
                temp_temp_list.append(token.idx - 1)
            temp_list.append(temp_temp_list)
            entity_list.append(temp_list)
        result = {}
        result["sentence"] = word_list
        result["entity"] = entity_list
        return result

    def __tokens_and_words_map(self, tokens, word_list):
        # create a mapping between tokens list and word list
        tokens_to_words = {}
        i = 0
        curr_token = ""
        for index, token in enumerate(tokens):
         
            if token == "[CLS]" or token == "[SEP]":
                tokens_to_words[index] = -1
                continue
            if token.startswith("##"):
                token = token.replace("##", "")
            curr_token += token
            if curr_token.casefold() == word_list[i].casefold():
                # clear the token
                curr_token = ""
                tokens_to_words[index] = i
                i += 1
            else:
                tokens_to_words[index] = i
        words_to_tokens = defaultdict(list)
        for i, v in tokens_to_words.items():
            words_to_tokens[v].append(i)
        return tokens_to_words, words_to_tokens

    def __processNerLabel(self, label_list, word_list):
        entities = []
        name_str = ""
        label_str = ""
        last_label_str = ""
        last_zero = False
        current_index_list = []
        for j in range(0, len(label_list)):
            if label_list[j] == 0:
                last_zero = True
                continue
            else:
                if label_list[j] % 2 == 1: # start with B
                    if len(name_str.strip()) > 0:
                        # append previous entity
                        if label_str == "":
                            label_str = last_label_str
                            name_str.lstrip(" ")
                        entities.append([name_str, label_str, current_index_list])
                        # reset name string
                        name_str = ""
                        current_index_list = []
                    if label_list[j] == 1:
                        label_str = "MISC"
                    elif label_list[j] == 3:
                        label_str = "PER"
                    elif label_list[j] == 5:
                        label_str = "ORG"
                    else:
                        label_str = "LOC"
                    current_index_list.append(j)
                    name_str += word_list[j]
                else: # start with I
                    if last_zero == True and name_str != "":
                        # append previous entity
                        if label_str == "":
                            label_str = last_label_str
                        entities.append([name_str, label_str, current_index_list])
                        # reset name string
                        name_str = ""
                        current_index_list = []
                    if last_zero == False and j > 0: name_str += " "
                    current_index_list.append(j)
                    name_str += word_list[j]
                    if label_list[j] == 2:
                        last_label_str = "MISC"
                    elif label_list[j] == 4:
                        last_label_str = "PER"
                    elif label_list[j] == 6:
                        last_label_str = "ORG"
                    else:
                        last_label_str = "LOC"
                    # print(name_str)
                last_zero = False
        if len(name_str.strip()) > 0: 
            if label_str == "": entities.append([name_str, last_label_str, current_index_list])
            else: entities.append([name_str, label_str, current_index_list])
        return entities

    def bert_ner(self, sentence):
        
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        # model = model.to(device)
        # tokenizer = tokenizer.to(device)
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(sentence)["input_ids"])

        word_list = sentence.split(" ")
        # pdb.set_trace()
        ner_results = nlp(sentence)
        result = {}
        tokens_to_words, words_to_tokens = self.__tokens_and_words_map(tokens, word_list)
        word_list_labels = [0] * len(word_list)
        ner_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        for ner_dict in ner_results:
            token_index = ner_dict["index"]
            word_index = tokens_to_words[token_index]
            current_label = ner_dict["entity"]
            word_list_labels[word_index] = ner_tags[current_label]
        result["sentence"] = word_list
        result["entity"] = self.__processNerLabel(word_list_labels, word_list)
        return result


    def stanza_ner(self, sentence, lang="en", processors= "tokenize,ner"):
        nlp = stanza.Pipeline(lang=lang, processors=processors)
        doc = nlp(sentence)
        

if __name__ == "__main__":
    apis = APIS()
    print(apis.bert_ner("In July , the former minister David Davis was one of nine senior Conservatives who wrote a letter to then Culture Secretary Nadine Dorries , warning the legal but harmful provision posed a threat to free speech ."))
    

