import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import nltk
import torch
import string
import numpy as np
from apis import APIS
from copy import deepcopy
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM
import random

'''
Automatically Repair NER errors raised by Testing part of TIN
'''
class Repairer:
    def __init__(self):
        self.null_reduction = 0.2
        self.subword_reduction = 0.5
        self.inconsistency_reduction = 0

    def bert_init(self):
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")
        self.bertori = BertModel.from_pretrained("bert-large-cased")
        self.bertmodel.eval().cuda()
        self.bertori.eval().cuda()

    def __bert_predict_mask(self, tokens, masked_index, number=50):
        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        prediction = self.bertmodel(tokens_tensor)
        topk = torch.topk(prediction[0][0], number)
        topk_idx = topk[1].tolist()
        topk_value = topk[0].tolist()
        res_tokens = self.berttokenizer.convert_ids_to_tokens(topk_idx[masked_index])
        res_values = topk_value[masked_index]
        return res_tokens, res_values

    def __mask_tokens(self, tokens, lindex, rindex):
        new_tokens = tokens[0:lindex] + ["[MASK]"] + tokens[rindex + 1:]
        new_tokens = ["[CLS]"] + new_tokens + ["[SEP]"]
        return new_tokens

    def __compEmbeddings(self, tokens, lindex, rindex):
        indexed_tokens = self.berttokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_tensors = torch.tensor([segments_ids]).cuda()
        with torch.no_grad():
            outputs = self.bertori(tokens_tensor, segments_tensors, output_hidden_states=True)
            hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            token_vecs_cat.append(cat_vec)
        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)

        ans_vec = deepcopy(token_vecs_sum[lindex])

        for index in range(lindex + 1, rindex + 1):
            ans_vec = ans_vec + token_vecs_sum[index]

        ans_vec = ans_vec / (rindex - lindex + 1)
        return torch.unsqueeze(ans_vec, dim=0)

    def __compute_simlarity(self, tokens, pert_tokens, lindex1, rindex1, lindex2, rindex2):

        # begin cav
        tokens_dp = deepcopy(tokens)
        pert_tokens_dp = deepcopy(pert_tokens)
        ha = self.__compEmbeddings(tokens_dp, lindex1, rindex1)
        hb = self.__compEmbeddings(pert_tokens_dp, lindex2, rindex2)

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return float(cos(ha, hb))

    def __to_possiblity(self, items):
        sum = 0
        result = []
        for item in items: sum += item[1]
        for item in items: result.append((item[0], item[1] / sum))
        return result

    def __consist_format(self, word1, word2):
        if ("#" in word1) and ("#" not in word2): return False
        if ("#" not in word1) and ("#" in word2): return False
        word1 = word1.replace("#", "")
        word2 = word2.replace("#", "")
        if (word1.isupper()) and (word2[0].isalpha() and word2[0].islower()): return False
        if (word1[0].isalpha() and word1[0].isupper()) and (
                word2[0].isalpha() and (not word2[0].isupper())): return False
        word1 = word1.replace(",", "")
        word1 = word1.replace(".", "")
        word2 = word2.replace(",", "")
        word2 = word2.replace(".", "")
        if (word1.isdigit()) and (not word2.isdigit()): return False
        if (not word1.isdigit()) and (word2.isdigit()): return False
        return True

    def __evaluation_function(self, bert_prob, similarity):
        return np.exp(2.5 * similarity) * bert_prob

    def repair(self, sentence, ori_prediction, suspicous_words, api, \
               threshold=5.5, Sthreshold=0.45):
        tokens = self.berttokenizer.tokenize(sentence)
        fixed_result = []
        for item in ori_prediction:
            if item[0] in suspicous_words: continue
            removed = False
            for word in suspicous_words:
                if item[0] in word or word in item[0]:
                    removed = True
            if not removed: fixed_result.append(item)
        Fixed_result = []
        for sus_word in suspicous_words:
            indexset = []
            lindex, rindex = -1, -1
            for li in range(0, len(tokens)):
                for ri in range(li, len(tokens)):
                    if self.berttokenizer.convert_tokens_to_string(tokens[li:ri + 1]).replace(" ", "") \
                            == sus_word.replace(" ", ""):
                        indexset.append([li, ri])

            Basic_tokens = sus_word.split(" ")
            for curindex in indexset:
                lindex, rindex = curindex[0], curindex[1]
                dict = {}
                dict["NULL"] = 0
                for basic_token in Basic_tokens:
                    if basic_token in string.punctuation: continue
                    ii, jj = -1, -1
                    for i in range(lindex, rindex + 1):
                        for j in range(i, rindex + 1):
                            if self.berttokenizer.convert_tokens_to_string(tokens[i:j + 1]) == basic_token:
                                ii, jj = i, j
                    if ii == -1: continue
                    new_tokens = self.__mask_tokens(tokens, ii, jj)
                    similar_words, values = self.__bert_predict_mask(new_tokens, ii + 1)

                    for word_index in range(0, len(similar_words)):
                        word = similar_words[word_index]
                        value = values[word_index]

                        if ii < jj and value < threshold:
                            continue
                        elif ii == jj and value < threshold:
                            continue

                        pert_tokens = deepcopy(tokens[0:ii + 1] + tokens[jj + 1:])
                        pert_tokens[ii] = word

                        delta = 1 - (jj - ii + 1)

                        similarity = self.__compute_simlarity(tokens, pert_tokens, \
                                                              lindex, rindex, lindex, rindex + delta)

                        if lindex < rindex and similarity < Sthreshold:
                            continue
                        elif lindex == rindex and similarity < Sthreshold:
                            continue

                        inconsistency = False
                        if not self.__consist_format(basic_token, word):
                            # inconsistency = True
                            continue

                        pert_tokens[ii] = word
                        request_sentence = self.berttokenizer.convert_tokens_to_string(pert_tokens)
                        pert_result = api(request_sentence)
                        sus_word_exist = False
                        check_word = self.berttokenizer.convert_tokens_to_string(pert_tokens[lindex:rindex + delta + 1])
                        for item in pert_result:
                            if item["text"] != check_word: continue
                            sus_word_exist = True
                            entity = item["entity"]
                            if entity not in dict: dict[entity] = 0
                            if inconsistency:
                                dict[entity] += self.inconsistency_reduction * \
                                                self.__evaluation_function(value, similarity)
                            else:
                                dict[entity] += self.__evaluation_function(value, similarity)
                            break
                        if not sus_word_exist:
                            if inconsistency:
                                dict["NULL"] += self.inconsistency_reduction * self.null_reduction * \
                                                self.__evaluation_function(value, similarity)
                            else:
                                dict["NULL"] += self.null_reduction * self.__evaluation_function(value, similarity)

                for index in range(lindex, rindex + 1):
                    if tokens[index] in Basic_tokens: continue
                    if "#" in tokens[index] and len(tokens[index]) <= 4: continue
                    if len(tokens[index]) <= 2: continue
                    if not (index >= lindex and index <= rindex): continue
                    new_tokens = self.__mask_tokens(tokens, index, index)
                    similar_words, values = self.__bert_predict_mask(new_tokens, index + 1)

                    for word_index in range(0, len(similar_words)):
                        word = similar_words[word_index]
                        value = values[word_index]
                        if value < threshold: continue
                        pert_tokens = deepcopy(tokens)
                        pert_tokens[index] = word
                        similarity = self.__compute_simlarity(tokens, pert_tokens, lindex, rindex, lindex, rindex)

                        if lindex < rindex and similarity < Sthreshold:
                            continue
                        elif lindex == rindex and similarity < Sthreshold:
                            continue

                        inconsistency = False
                        if not self.__consist_format(tokens[index], word):
                            continue

                        request_sentence = self.berttokenizer.convert_tokens_to_string(pert_tokens)
                        pert_result = api(request_sentence)
                        sus_word_exist = False
                        check_word = self.berttokenizer.convert_tokens_to_string(pert_tokens[lindex:rindex + 1])
                        for item in pert_result:
                            if item["text"] != check_word: continue
                            sus_word_exist = True
                            entity = item["entity"]
                            if entity not in dict: dict[entity] = 0
                            if inconsistency:
                                dict[entity] += self.subword_reduction * self.inconsistency_reduction * \
                                                self.__evaluation_function(value, similarity)
                            else:
                                dict[entity] += self.subword_reduction * self.__evaluation_function(value, similarity)
                            break
                        if not sus_word_exist:
                            if inconsistency:
                                dict["NULL"] += self.null_reduction * self.subword_reduction * \
                                                self.inconsistency_reduction * \
                                                self.__evaluation_function(value, similarity)
                            else:
                                dict["NULL"] += self.null_reduction * self.subword_reduction * \
                                                self.__evaluation_function(value, similarity)

                items = sorted(dict.items(), key=lambda x: x[1], reverse=True)
                if items[0][1] > 0:
                    items = self.__to_possiblity(items)
                    # print("#", sus_word, items)
                    if (items[0][0] != "NULL"):
                        Fixed_result.append([[sus_word] + [items[0][0]], dict[items[0][0]], lindex, rindex])
                else:
                    for item in ori_prediction:
                        if item[0] == sus_word:
                            Fixed_result.append([item, 1e18, lindex, rindex])
                            break
        for x in Fixed_result:
            flag_remove = 0
            for y in Fixed_result:
                xlindex, xrindex = x[2], x[3]
                ylindex, yrindex = y[2], y[3]
                xconf, yconf = x[1], y[1]
                if xlindex == ylindex and xrindex == yrindex: continue
                if (xlindex >= ylindex and xlindex <= yrindex) or (ylindex >= xlindex and ylindex <= xrindex):

                    if xconf < yconf:
                        flag_remove = 1
                    elif xconf == yconf and len(x[0][0]) > len(y[0][0]):
                        flag_remove = 1
            if not flag_remove: fixed_result.append(x[0])
        ''' 
        intermediate output info
        '''
        print("============================================================")
        print("Suspicious Sentence: ", sentence)
        print("Fixed NER Prediction: ", fixed_result)
        print("============================================================")
        return fixed_result

    def __get_suspicous_words(self, sentence, pert_sentence, ori_prediction, pert_prediction, sus):
        sus_words = set(sus)
        ori_dict = {}
        pert_dict = {}
        for entity in ori_prediction:
            if entity[0] not in ori_dict.keys():
                ori_dict[entity[0]] = []
            ori_dict[entity[0]].append(entity[1])
        for entity in pert_prediction:
            if entity[0] not in pert_dict.keys():
                pert_dict[entity[0]] = []
            pert_dict[entity[0]].append(entity[1])
        for item in ori_dict.items():
            if item[0] not in pert_sentence: continue
            if item[0] not in pert_dict.keys():
                sus_words.add(item[0])
            elif pert_dict[item[0]] != item[1]:
                sus_words.add(item[0])
        for item in pert_dict.items():
            if item[0] not in sentence: continue
            if item[0] not in ori_dict.keys():
                sus_words.add(item[0])
            elif ori_dict[item[0]] != item[1]:
                sus_words.add(item[0])
        for item in ori_dict.items():
            if item[0] in pert_sentence: continue
            if not item[0][0].isalpha(): continue
            text = item[0][0].swapcase() + item[0][1:]
            if text not in pert_sentence: continue
            if text not in pert_dict.keys():
                sus_words.add(item[0])
            elif pert_dict[text] != item[1]:
                sus_words.add(item[0])

        for item in pert_dict.items():
            if item[0] in sentence: continue
            if not item[0][0].isalpha(): continue
            text = item[0][0].swapcase() + item[0][1:]
            if text not in sentence: continue
            if text not in ori_dict.keys():
                sus_words.add(text)
            elif ori_dict[text] != item[1]:
                sus_words.add(text)
        result = []
        for x in sus_words: result.append(x)
        return result

    def __repair_sus(self, ori_sentence, pert_sentence, ori_prediction, pert_prediction, ori_sus, pert_sus, api):
        ori_sus_words = self.__get_suspicous_words(ori_sentence, pert_sentence, \
                                                   ori_prediction, pert_prediction, ori_sus)
        pert_sus_words = self.__get_suspicous_words(pert_sentence, ori_sentence, \
                                                    pert_prediction, ori_prediction, pert_sus)
        print("Suspicious Entities: ", ori_sus_words)
        ori_fixed_result = self.repair(ori_sentence, ori_prediction, ori_sus_words, api)
        pert_fixed_result = self.repair(pert_sentence, pert_prediction, pert_sus_words, api)
        return ori_fixed_result, pert_fixed_result, ori_sus_words, pert_sus_words

    def repair_suspicious_issue(self, input_file, output_file, apis):
        with open(input_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        g = open(output_file, "w", encoding="utf-8")
        result = []
        for dict in data:
            ori_sus_word, pert_sus_word = [], []
            if "sus_words" in dict.keys():
                ori_sus_word = [dict["sus_words"][0]]
                pert_sus_word = [dict["sus_words"][1]]
            ori_fixed_result, pert_fixed_result, ori_sus_words, pert_sus_words = \
                self.__repair_sus(dict["original"]["sentence"], \
                                        dict["new"]["sentence"], \
                                        dict["original"]["entity"], \
                                        dict["new"]["entity"], ori_sus_word, pert_sus_word, apis)
            fix_ori, fix_new = deepcopy(dict["original"]), deepcopy(dict["new"])
            fix_ori["entity"], fix_new["entity"] = ori_fixed_result, pert_fixed_result
            result.append({"original": fix_ori, "new": fix_new})
        json.dump(result, g)



if __name__ == "__main__":
    # For example:
    apis = APIS()
    NER_repairer = Repairer()
    NER_repairer.bert_init()
    NER_repairer.repair_suspicious_issue("./data/suspicious_flair.json",
                            "./data/suspicious_flair_repair_json", apis.flair)