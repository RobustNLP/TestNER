# to authenticate the API: use the following command:
import os
import time
import boto3
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from flair.data import Sentence
from flair.models import SequenceTagger


class APIS:
    def aws(self, sentence, comprehend_client = boto3.client('comprehend')):
        entities = comprehend_client.detect_entities(Text = sentence, LanguageCode = 'en')
        result = []
        for entity in entities["Entities"]:
            dict = {}
            dict["text"] = entity["Text"]
            dict["entity"] = entity["Type"]
            result.append(dict)
        return result

    def authenticate_client(self):
        key = "xxx"
        endpoint = "xxx"
        ta_credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(
                endpoint=endpoint, 
                credential=ta_credential)
        return text_analytics_client

    def azure(self, sentence):
        client = self.authenticate_client()
        connected = False
        while not connected:
            time.sleep(0.01)
            try:
                # entity_info = {}
                entity_set = set()
                sentence_list = []
                sentence_list.append(sentence)
                reponse = client.recognize_entities(documents = sentence_list)[0]
                result = []
                for entity in reponse.entities:
                    dict = {}
                    dict["text"] = entity.text
                    dict["entity"] = entity.category
                    result.append(dict)
                    # entity_info[entity.text] = entity.category
                connected = True
                return result
            
            except Exception as err:
                connected = False
                print("Encountered exception. {}".format(err))
    # def flair(self, sentence, tagger = SequenceTagger.load('flair/ner-english-ontonotes-large')):
    # def flair(self, sentence, tagger = SequenceTagger.load('flair/ner-english-large')):
    def flair(self, sentence, tagger = SequenceTagger.load('flair/ner-english-large')):
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
            temp_list.append(entity.get_label('ner').value)
            # entity indexes
            temp_temp_list = []
            for token in entity:
                temp_temp_list.append(token.idx - 1)
            temp_list.append(temp_temp_list)
            entity_list.append(temp_list)
        result = []
        for entity in entity_list:
            dict = {}
            dict["text"] = entity[0]
            dict["entity"] = entity[1]
            result.append(dict)
        return result
        #return word_list, entity_list

    def flair_OntoNotes(self, sentence, tagger = SequenceTagger.load('flair/ner-english-ontonotes-large')):
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
            temp_list.append(entity.get_label('ner').value)
            # entity indexes
            temp_temp_list = []
            for token in entity:
                temp_temp_list.append(token.idx - 1)
            temp_list.append(temp_temp_list)
            entity_list.append(temp_list)
        result = []
        for entity in entity_list:
            dict = {}
            dict["text"] = entity[0]
            dict["entity"] = entity[1]
            result.append(dict)
        return result
        #return word_list, entity_list
if __name__ == "__main__":
    apis = APIS()