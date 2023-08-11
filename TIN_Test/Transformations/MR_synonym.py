import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import spacy
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import *
import itertools
import random
import sys
import pdb
import ipdb
# import the AEON, please change the follosing path with the path ofaeon.py
# sys.path.insert(1, '$HOME$/NER_MRs/MRs/AEON/utils')
from aeon import *
from copy import deepcopy
from collections import defaultdict
import torch
import itertools
import nltk
from transformers import BertConfig, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, BertForMaskedLM
import pathlib

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    return_list = []
    for s in sentences:
        if len(s.replace(" ", "")) != 0:
            return_list.append(s)
    return return_list

class MR_synonym:
    def MR_init(self, aeon_flag = True, evaluate = False, aeon_threshold = 0.01):
        self.nlp = spacy.load("en_core_web_sm")
        self.s2v = self.nlp.add_pipe("sense2vec")
        self.s2v.from_disk(str(pathlib.Path().absolute()) + "/s2v_old")
        # self.s2v.from_disk("../s2v_old") # you path to the s2v package
        self.nlp_stanford = StanfordCoreNLP(
            str(pathlib.Path().absolute())+'/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
        # self.nlp_stanford = StanfordCoreNLP(
        #     r'../corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
        self.scorer_default = Scorer(
            8, 
            'bert-base-uncased', 
            'princeton-nlp/sup-simcse-bert-base-uncased',
            True
        )
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        self.bertmodel = BertForMaskedLM.from_pretrained("bert-large-cased")
        self.bertori = BertModel.from_pretrained("bert-large-cased")
        self.bertmodel.eval().cuda()
        self.bertori.eval().cuda()
        self.threshold = 0.65 # threshold for similarity (when generating)
        self.similarity_threshold = 0.65
        self.max_sentences = 5 # maximum number of similar sentences
        self.AEON_threshold = aeon_threshold
        self.bound = 10 # max synonym to choose
        self.random_seed = 14
        self.filter_flag = aeon_flag
        self.evaluate = evaluate
        
    def MR_close(self):
        self.nlp_stanford.close()
    
    def __check_normal_string(self, word):
        def find_url(string):
            regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
            url = re.findall(regex, string)
            return [x[0] for x in url]
        
        for character in word:
            if ord(character) not in range(32, 128):
                return False
        if find_url(word):
            return False
        if any(item in word for item in ["#", "/", "^", "@", "<", ">"]):
            return False
        return True
        
    
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
            if curr_token.replace(" ", "") == word_list[i].replace(" ", ""):
                # if curr_token.casefold() == word_list[i].casefold():
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

    def __node_to_index(self, root, node_reference):
        # get the start index of the tree node given the
        count = 0
        for depth in reversed(range(len(node_reference))):
            current_index = node_reference[depth]
            for layer_index in range(current_index):
                # refer to the node by root
                root_name = "root"
                for i in range(depth):
                    root_name += "[" + str(node_reference[i]) + "]"
                root_name += "[" + str(layer_index) + "]"
                eval_str = ".leaves()"
                count += len(eval(root_name + eval_str))
        return count
    
    def __find_NP_nodes(self, root):
        node_list = []
        nodeIndex_list = []

        def findNP(node):
            found = False
            search_list = []
            if type(node) == str:
                return False
            for child in node:
                search_list.append(child)
            while len(search_list) > 0:
                curr = search_list.pop(0)
                if type(curr) == str: # leaf
                    continue
                else:
                    if curr.label().startswith("NP"):
                        found = True
                        return found
                    else:
                        for child in curr:
                            search_list.append(child)
            return found

        def findNodes_NP(node): # use DFS search
            find_list = []
            find_list.append((node, [])) # [0]-> record the indexing info of the current node
            while len(find_list) > 0:
                curr = find_list.pop(0)
                if type(curr[0]) == str: # leaf
                    continue
                else:
                    if curr[0].label().startswith("NP"):
                        if findNP(curr[0]) == False:
                            node_list.append(curr[0])
                            nodeIndex_list.append(curr[1])
                    i = 0
                    for child in curr[0]:
                        temp_list = curr[1].copy()
                        temp_list.append(i)
                        find_list.append((child, temp_list))
                        i += 1
        
        findNodes_NP(root)
        return node_list, nodeIndex_list
    
    def __changeNP(self, node, NEs):
        NEs = [e.replace(" ", "") for e in NEs]
        # check if the current node contains the entity
        curr_word = ("".join(node.leaves())).replace(" ", "")
        curr_word_check = (" ".join(node.leaves())).lstrip().rstrip()
        # print("current check word is: ", curr_word_check)
        if any(curr_word_check.casefold().startswith(item) for item in ["the ", "a ", "an "]):
            curr_word_check = (" ".join(curr_word_check.split(" ")[1:])).replace(" ","")
        else:
            curr_word_check = curr_word_check.replace(" ", "")
        if any(ext in curr_word_check for ext in NEs):
            return [node]
        if curr_word_check.casefold() not in ["a", "the", "an"] and any(curr_word_check in ext for ext in NEs):
            return [node]
        if curr_word_check.casefold() in ["a", "the", "an"]:
            return [node]
        
        # check the leftest node
        is_a = False
        keep_left = False
        temp_node = node.copy() # a temp node to find the noun    
        if temp_node.leaves()[0].casefold() == "a" or temp_node.leaves()[0].casefold() == "an":
            is_a = True
        if temp_node[0].label().startswith("DT") or temp_node[0].label().startswith("PRP"):
            temp_node.pop(0)
            keep_left = True
        word_list = temp_node.leaves();
        sub_string = " ".join(word_list)
        # use the sense2vec to substitute
        doc = self.nlp(sub_string)
        try:
            freq = doc[0:len(doc)]._.s2v_freq
            vector = doc[0:len(doc)]._.s2v_vec
            generate_list = []
            most_similar = doc[0:len(doc)]._.s2v_most_similar(self.bound)
        except:
            return [node]
        for index, info in enumerate(most_similar):
            word = info[0][0]
            pos = info[0][1]
            score = info[1]
            if not self.__check_normal_string(word):
                continue
            if pos != "NOUN":
                continue
            if score < self.threshold:
                continue
            word = word.lstrip().rstrip()
            if word.replace(" ", "").casefold() == sub_string.replace(" ", "").casefold():
                continue
            if keep_left:
                if is_a and word.endswith("s"):
                    continue
                # check if the word start with DT
                if word.casefold().startswith('the '):
                    word = word[4:]
                if word.casefold().startswith('an '):
                    word = word[3:]
                if word.casefold().startswith('a '):
                    word = word[2:]
                curr_node = Tree("NP", [Tree("NP", [word])])
                curr_node.insert(0, node[0])
                generate_list.append(curr_node)
            else:
                curr_node = Tree("NP", [word])
                generate_list.append(curr_node)
        if len(generate_list) == 0: generate_list = [temp_node]
        return generate_list
    
    
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
   
    
    def __generate_syn_sentences_single(self, sentence, NEs):
        filtered_number = 0
        # ipdb.set_trace()
        parsingTree = Tree.fromstring(self.nlp_stanford.parse(sentence))
        # parsingTree.pretty_print()
        parsingTree_repr = repr(parsingTree)
        node_list, nodeIndex_list = self.__find_NP_nodes(parsingTree)
        substitution_dict = {}
        changable_list = []
        for node in node_list:
            substitution_dict[repr(node)] = self.__changeNP(node, NEs)
        for node in node_list:
            changable_list.append([i for i in range(len(substitution_dict[repr(node)]))])
        node_reprs = [repr(node) for node in node_list]
        original_words = [" ".join(node.leaves()) for node in node_list]
        result_sentences = []
        new_words = []
        for change_index in range(len(changable_list)):
            start_index = self.__node_to_index(eval(parsingTree_repr), nodeIndex_list[change_index])
            
            for candidate_index in range(len(changable_list[change_index])):
                # map the node repr to the current index of candidate word to change
                new_words_current = []
                temp_sent = eval(parsingTree_repr)
                node_repr = node_reprs[change_index] # get the current node repr of the node to be changed
                node_new = substitution_dict[node_repr][candidate_index]
                new_word = " ".join(node_new.leaves())
                if original_words[change_index] == new_word:
                    continue
                new_words_current.append(new_word)
                # use exec to change the subtree of the current tree
                node_name = "temp_sent"
                assign_name = "=node_new"
                for j in range(len(nodeIndex_list[change_index])):
                    node_name += "[" + str(nodeIndex_list[change_index][j])+ "]"
                ori_word = " ".join(eval(node_name).leaves())
                ori_offset = len(eval(node_name).leaves())
                ori_word_list = temp_sent.leaves()
                exec(node_name + assign_name)
                new_word = " ".join(node_new.leaves())
                new_offset = len((" ".join(node_new.leaves())).split(" "))
                node_new_list = []
                for leaf in node_new.leaves():
                    node_new_list.extend(leaf.split(" "))
                new_word_list = temp_sent.leaves()[:start_index] + node_new_list + temp_sent.leaves()[start_index + new_offset:]
                # ipdb.set_trace()
                # ori_np = original_words[change_index]
                # new_np = new_word
                ori_bert_tokens = self.berttokenizer.tokenize(" ".join(ori_word_list))
                new_bert_tokens = self.berttokenizer.tokenize(" ".join(new_word_list))
                token2word_ori, word2token_ori = self.__tokens_and_words_map( ori_bert_tokens, ori_word_list )
                token2word_new, word2token_new = self.__tokens_and_words_map( new_bert_tokens, new_word_list )
                
                start_index_token_ori = word2token_ori[start_index][0]
                end_index_ori = start_index + ori_offset
                if end_index_ori >= len(ori_word_list):
                    filtered_number += 1
                    continue
                end_index_token_ori = word2token_ori[end_index_ori][len(word2token_ori[end_index_ori])-1]

                start_index_token_new = word2token_new[start_index][0]
                end_index_new = start_index + new_offset
                if end_index_new >= len(new_word_list):
                    filtered_number += 1
                    continue
                end_index_token_new = word2token_new[end_index_new][len(word2token_new[end_index_new])-1]
                
                # compare the embedding similarity, if the similarity is greater than a threshold, continue
                similarity = self.__compute_simlarity(ori_bert_tokens, new_bert_tokens, start_index_token_ori, end_index_token_ori, start_index_token_new, end_index_token_new)
                # print(similarity.item())
                # ipdb.set_trace()
                if similarity.item() <= self.similarity_threshold:
                    filtered_number += 1
                    continue
                
                return_tokens = temp_sent.leaves()
                for i in range(0, len(return_tokens)):
                    if return_tokens[i] == "-LRB-":
                        return_tokens[i] = "("
                    elif return_tokens[i] == "-RRB-":
                        return_tokens[i] = ")"
                    
                return_string = " ".join(return_tokens)
                if " - " not in sentence: return_string = return_string.replace(" - ", "-")
                
                result_sentences.append(return_string)
                new_words.append(new_words_current)
        if self.evaluate == True:
            return result_sentences, new_words, filtered_number
        return result_sentences, new_words
    
    def generate_syn_sentences(self, sentence, NEs):
        ori_number = 0
        sim_number = 0
        aeon_number = 0
        sentence_list = split_into_sentences(sentence)
        new_indexes_list = [] # list of index list to repr the new sentence to choose
        new_sentences_list = [] # list of list of new sentences
        new_words_list = [] # list of list of new words
        total_filtered = 0
        for sent in sentence_list:
            new_indexes = []
            new_sentences = []
            new_words = []
            if self.evaluate == True:
                new_sentences_current, new_words_current, filter_count = self.__generate_syn_sentences_single(sent, NEs)
                for index, new_sentence in enumerate(new_sentences_current):
                    total_filtered += filter_count
                    new_indexes.append(index)
                    new_sentences.append(new_sentence)
                    new_words.append(new_words_current[index])
                new_indexes_list.append(new_indexes)
                new_sentences_list.append(new_sentences)
                new_words_list.append(new_words)
                continue
            new_sentences_current, new_words_current = self.__generate_syn_sentences_single(sent, NEs)
            for index, new_sentence in enumerate(new_sentences_current):
                new_indexes.append(index)
                new_sentences.append(new_sentence)
                new_words.append(new_words_current[index])
            new_indexes_list.append(new_indexes)
            new_sentences_list.append(new_sentences)
            new_words_list.append(new_words)
        result_sentences = []
        for indexes in itertools.product(*new_indexes_list):
            result_sentence_list = []
            result_new_words_list = []
            for pos, index in enumerate(indexes):
                result_sentence_list.append(new_sentences_list[pos][index].lstrip(" ").rstrip(" "))
                result_new_words_list.extend(new_words_list[pos][index])
            result_sentence = " ".join(result_sentence_list)
            result_sentences.append((result_sentence, result_new_words_list))
        sim_number = len(result_sentences) 
        ori_number = sim_number + total_filtered
        # naturalness filter
        return_sentences = []
        if self.filter_flag:
            ori_score = self.__get_AEON_score(sentence)
            for (sent, new_words) in result_sentences:
                new_score = self.__get_AEON_score(sent)
                if new_score >= ori_score - self.AEON_threshold:
                    return_sentences.append((sent, new_words))
        else:
            return_sentences = result_sentences
        aeon_number = len(return_sentences)
                
        # return result
        changed = False
        if len(return_sentences) == 0 or \
            (len(return_sentences) == 1 and return_sentences[0][0].replace(" ","") == sentence.replace(" ", "")):
                changed = False
                if self.evaluate == True:
                    return [ori_number, sim_number, aeon_number]
                return (changed, [])
        else:
            changed = True
            if self.evaluate == True:
                return [ori_number, sim_number, aeon_number]
            return (changed, return_sentences)
        
    def show(self, sentence):
        self.nlp = StanfordCoreNLP(
            r'$HOME$/NER_MRs/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
        parsingTree = Tree.fromstring(self.nlp.parse(sentence))
        parsingTree.pretty_print()    
                
if __name__ == "__main__":
    mr_synonym = MR_synonym()
    mr_synonym.MR_init()
    print(mr_synonym.generate_syn_sentences("In July , the former minister David Davis was one of nine senior Conservatives who wrote a letter to then Culture Secretary Nadine Dorries , warning the legal but harmful provision posed a threat to free speech .", ['David Davis', 'Conservatives', 'Nadine Dorries']))
    mr_synonym.MR_close() 