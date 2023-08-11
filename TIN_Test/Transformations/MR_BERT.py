import json
import os
import ipdb
import nltk
import torch
import numpy as np
from copy import deepcopy
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM
from collections import defaultdict
import itertools
import random
import sys
# import the AEON, please change the follosing path with the path ofaeon.py
sys.path.insert(1, '$HOME$/NER_MRs/MRs/AEON/utils')
from aeon import *

# I use some of Joye;s codes
class MR_BERT:
    def MR_init(self, aeon_flag = True, evaluate = False, aeon_threshold = 0.02):
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")
        self.bertori = BertModel.from_pretrained("bert-large-cased")
        self.bertmodel.eval().cuda()
        self.bertori.eval().cuda()
        # self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
        self.poses = ["JJ", "VB"]
        self.scorer_default = Scorer(
            8, 
            'bert-base-uncased', 
            'princeton-nlp/sup-simcse-bert-base-uncased',
            True
        )
        self.similarity_threshold = 0.65
        self.predcition_threshold = 9
        self.max_sentences = 5
        # self.AEON_threshold = 0
        self.AEON_threshold = aeon_threshold
        self.rand_seed = 14
        self.evaluate = evaluate
        self.filter_flag = aeon_flag
    
    def __get_entity_ids(self, NEs, word_list):
        """
        Find the ids in the word list of the entities

        return:
        a list of ids of the entities
        """
        entity_ids = []
        for entity_name in NEs:
            # change eneity words into strings without whitespace
            curr_name = entity_name.replace(" ", "")
            ori_curr_name = '%s' % curr_name
            id_list = []
            for index, word in enumerate(word_list):
                word = word.replace(" ", "")
                if curr_name.startswith(word):
                    curr_name = curr_name[len(word):]
                    id_list.append(index)
                else:
                    curr_name = ori_curr_name
                    id_list = [] # clear the ids
                if len(curr_name) == 0:
                    for i in id_list:
                        if i not in entity_ids:
                            entity_ids.append(i)
                    curr_name = ori_curr_name
        return entity_ids
    
    def __get_AEON_score(self, sentence):
        return self.scorer_default.compute_naturalness(sentence)
    
    def __tokens_and_words_map(self, tokens, word_list):
        # create a mapping between tokens list and word list
        tokens_to_words = {}
        i = 0
        curr_token = ""
        for index, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                tokens_to_words[index] = -1
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
    
    def __bert_predict_mask(self, tokens, masked_index, number = 35):
        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        prediction = self.bertmodel(tokens_tensor)
        topk = torch.topk(prediction[0][0], number)
        topk_idx = topk[1].tolist()
        topk_value = topk[0].tolist()
        res_tokens = self.berttokenizer.convert_ids_to_tokens(topk_idx[masked_index])
        res_values = topk_value[masked_index]
        # ipdb.set_trace()
        return res_tokens, res_values
        
    def __mask_tokens(self, tokens, lindex, rindex):
        new_tokens = tokens[0:lindex] + ["[MASK]"] + tokens[rindex + 1:]
        new_tokens = ["[CLS]"] + new_tokens + ["[SEP]"]
        # print(new_tokens)
        return new_tokens

    # compute the word embedding activated
    def __compEmbeddings( self, tokens, lindex, rindex ):
        rindex = rindex - 1
        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)
        tokens_tensor = torch.tensor( [indexed_tokens] ).cuda()
        segments_tensors = torch.tensor( [segments_ids] ).cuda()
        with torch.no_grad():
            outputs = self.bertori( tokens_tensor, segments_tensors, output_hidden_states=True )
            hidden_states = outputs[2]
        
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)
        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []
        # `token_embeddings` is a [22 x 12 x 768] tensor.
        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            # Concatenate the vectors (that is, append them together) from the last 
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)
            
        # Stores the token vectors, with shape
        token_vecs_sum = []
        # For each token in the sentence...
        for token in token_embeddings:
            
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)
        
        ans_vec = deepcopy(token_vecs_sum[lindex])
        for index in range(lindex + 1, rindex + 1):
            ans_vec = ans_vec + token_vecs_sum[index]
        
        ans_vec = ans_vec / (rindex - lindex + 1)
        # return the word embeddings for the word
        return torch.unsqueeze(ans_vec, dim=0 )
    
    
    def __compute_simlarity(self, tokens, pert_tokens, lindex1, rindex1, lindex2, rindex2):

        # begin cav
        tokens_dp = deepcopy(tokens)
        pert_tokens_dp = deepcopy(pert_tokens)
        ha = self.__compEmbeddings( tokens_dp, lindex1, rindex1 )
        hb = self.__compEmbeddings( pert_tokens_dp, lindex2, rindex2 )
        
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return abs(cos(ha, hb))
    
    
    def __to_possiblity(self, items):
        sum = 0
        result = []
        for item in items: sum += item[1]
        for item in items: result.append((item[0], item[1] / sum))
        return result
    
    def generate_sentences(self, sentence, NEs):
        tokens = self.berttokenizer.tokenize(sentence)
        word_list = sentence.split(" ")
        tokens_to_words, words_to_tokens = self.__tokens_and_words_map(tokens, word_list)
        # select the words that can be perturbed
        perturbable_ids = []
        entity_ids = self.__get_entity_ids(NEs, word_list)
        pos_word_list = nltk.pos_tag(tokens)
        for id_token, pos_info in enumerate(pos_word_list):
            id_word = tokens_to_words[id_token]
            if id_word in entity_ids:
                continue
            if pos_info[1] in self.poses:
                if pos_info[1] == "JJ" and (id_word+1) in entity_ids:
                    # do not change the adj that is adjacent to the entity
                    continue
                if id_word not in perturbable_ids: # prevent repitition
                    perturbable_ids.append(id_word)
        substitution_dict = defaultdict(list)
        ori_number = 0
        sim_number = 0
        # generate new sentences
        for word_id in perturbable_ids:
            token_ids = words_to_tokens[word_id]
            lindex = token_ids[0]
            rindex = token_ids[len(token_ids)-1] + 1
            new_tokens = self.__mask_tokens(tokens, lindex, rindex)
            res_tokens, res_values = self.__bert_predict_mask(new_tokens, lindex+1)
            lindex2 = lindex
            rindex2 = lindex + 1
            # compute cosine similarity of
            for index, token in enumerate(res_tokens[:10]):
                # fliter out the illegal format
                if token.startswith("##"):
                    continue
                # check the prediction score
                if res_values[index] <= self.predcition_threshold:
                    continue
                # compare pos tag, if the tag is not consistent, abort
                pert_tokens = tokens[0: lindex] + [token] + tokens[rindex: ]
                pos_pert_token = nltk.pos_tag(pert_tokens)
                
                if pos_word_list[lindex][1] != pos_pert_token[lindex][1]:
                    continue
                # compare the embedding similarity, if the similarity is greater than a threshold, continue
                similarity = self.__compute_simlarity(tokens,pert_tokens, lindex, rindex, lindex2, rindex2)
                ori_number += 1
                if similarity.item() <= self.similarity_threshold:
                    continue
                sim_number += 1
                if token != word_list[tokens_to_words[lindex]]:
                    substitution_dict[tokens_to_words[lindex]].append(token)
        
        # change the words according to the substitution dict
        candidate_sentences = []
        for (id, candidate_words) in substitution_dict.items():
            for candidate_word in candidate_words:
                temp_word_list = word_list.copy()
                temp_word_list[id] = candidate_word
                new_sentence = " ".join(temp_word_list)
                new_words = [candidate_word]
                candidate_sentences.append((new_sentence, new_words))
        
        # fitler the candiate sentences by AEON
        filtered_sentences = []
        if self.filter_flag:
            ori_score = self.__get_AEON_score(sentence)
            for index, candidate in enumerate(candidate_sentences):
                new_score = self.__get_AEON_score(candidate[0])
                if new_score >= ori_score - self.AEON_threshold:
                    filtered_sentences.append(candidate)
        else:
            filtered_sentences = candidate_sentences
        aeon_number = len(filtered_sentences)
        
        changed = False
        if len(filtered_sentences) == 0 or \
            (len(filtered_sentences) == 1 and filtered_sentences[0] == sentence):
                changed = False
                if self.evaluate == True:
                    return [ori_number, sim_number, aeon_number]
                return (changed, filtered_sentences)
        else:
            changed = True
            if self.evaluate == True:
                return [ori_number, sim_number, aeon_number]
            return (changed, filtered_sentences)
        
            
def format_to_string(result_dict):
    # convert the result dict to inputable string format
    sentence_str = " ".join(result_dict["sentence"])
    entities = result_dict["entity"]
    NEs = []
    for entity in entities:
        NEs.append(entity[0])
    return sentence_str, NEs         
        
if __name__ == "__main__":
    mr_bert = MR_BERT()
    mr_bert.MR_init()
    print(mr_bert.generate_sentences("He said : For me this is such an exciting opportunity for Cornwall and something I ' m now desperate to see happen .", ["Cornwall"]))    