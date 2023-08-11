import os
from Transformations.MR_BERT import *
from Transformations.MR_structure import *
from Transformations.MR_synonym import *
from Transformations.MR_intra_shuffle import *
from flair.models import SequenceTagger
from apis import *
import random
import json
import pdb
import ipdb
import copy
from nltk.tree import *
import argparse
import sys
sys.path.insert(1, 'AEON/utils')
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

def NLTK_tree_get_sentence(leaves):
    for i in range(0, len(leaves)):
        if leaves[i] == "-LRB-":
            leaves[i] = "("
        elif leaves[i] == "-RRB-":
            leaves[i] = ")"
    return_string = " ".join(leaves)
    return return_string

def format_to_string(result_dict):
    # convert the result dict to inputable string format
    sentence_str = " ".join(result_dict["sentence"])
    entities = result_dict["entity"]
    NEs = []
    for entity in entities:
        NEs.append(entity[0])
    return sentence_str, NEs

def format_to_nameTypeList(result_dict):
    # convert the result dict to [(name, type)] format
    entities = result_dict["entity"]
    entities = sorted(entities,key=lambda l:l[2][0])
    ent_list = []
    for entity in entities:
        ent_list.append((entity[0], entity[1]))
    return ent_list

def check_mr_intersection_new_words(sentence, pert_sentence, ori_prediction, pert_prediction):
    sus_words = set()
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
    result = []
    for x in sus_words: result.append(x)
    return len(result) > 0


# Function implemented by cav
def check_mr_new_words(list1, list2, new_words):
    sus_flag = False
    # missing classfication is not tolerable
    set1 = set(list1)
    set2 = set(list2)
    diff_2to1 = set2 - set1
    for ele in diff_2to1:
        if not any(ele[0] in new_word for new_word in new_words):
            sus_flag = True
 
    if (set2-diff_2to1) != set1:
        sus_flag == True
    
    if sus_flag:
        return False
    else:
        return True

def check_mr_equal(list1, list2):
    # we need to deal with some situations caused by punctuations
    list1_check = set()
    list2_check = set()
    for item in list1:
        list1_check.add((item[0].replace(" ", ""), item[1]))
    for item in list2:
        list2_check.add((item[0].replace(" ", ""), item[1]))
    return list1_check == list2_check

def run_flair(sentences, tag = "flair/ner-english-large", sava_name="bbc_flair_original.json"):
    apis = APIS()
    # step1: run the dataset on apis
    results = []
    tagger = SequenceTagger.load(tag)
    nlp = StanfordCoreNLP(
        str(pathlib.Path().absolute())+'/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
    # step1: run the dataset on apis
    results = []
    for sentence in sentences:
        sent = NLTK_tree_get_sentence(Tree.fromstring(nlp.parse(sentence)).leaves())
        print(sent)
        # pdb.set_trace()
        results.append(apis.flair(sent, tagger))
    nlp.close()
            
    with open(sava_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def run_azure(sentences, save_name="bbc_azure_original.json"):
    apis = APIS()
    # step1: run the dataset on apis
    results = []
    nlp = StanfordCoreNLP(
        str(pathlib.Path().absolute())+'/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
    # step1: run the dataset on apis
    results = []
    for sentence in sentences:
        sent = NLTK_tree_get_sentence(Tree.fromstring(nlp.parse(sentence)).leaves())
        print(sent)
        results.append(apis.azure(sent))
    nlp.close()
            
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

def run_aws(sentences, save_name="bbc_aws_original.json"):
    apis = APIS()
    # step1: run the dataset on apis
    results = []
    nlp = StanfordCoreNLP(
        str(pathlib.Path().absolute())+'/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
    # step1: run the dataset on apis
    results = []
    for sentence in sentences:
        sent = NLTK_tree_get_sentence(Tree.fromstring(nlp.parse(sentence)).leaves())
        print(sent)
        results.append(apis.aws(sent))
    nlp.close()
            
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

def run_bert_ner(sentences, save_name="conll_bert_ner_original.json"):
    apis = APIS()
    # step1: run the dataset on apis
    results = []
    nlp = StanfordCoreNLP(
        str(pathlib.Path().absolute())+'/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
    # step1: run the dataset on apis
    results = []
    for sentence in sentences:
        print(sentence)
        sents = split_into_sentences(sentence)
        new_sents = []
        for sent in sents:
            new_sent = NLTK_tree_get_sentence(Tree.fromstring(nlp.parse(sent)).leaves())
            new_sents.append(new_sent)
        # sent = NLTK_tree_get_sentence(Tree.fromstring(nlp.parse(sentence)).leaves())
        result_sent = " ".join(new_sents)
        print(result_sent)
        results.append(apis.bert_ner(result_sent))
    nlp.close()
            
    with open(save_name, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

def run_MR_BERT(file_name, aeon_flag, aeon_threshold=0.01):
    # original_results = json.load(open(file_name))[:100]
    original_results = json.load(open(file_name))
    mr_bert = MR_BERT()
    mr_bert.MR_init(aeon_flag=aeon_flag, aeon_threshold= aeon_threshold)
    mr_bert_result = []
    for index, sentence in enumerate(original_results):
        print(str(file_name), index)
        print(sentence)
        sentence_str, NEs = format_to_string(sentence)
        new_sentences = mr_bert.generate_sentences(sentence_str, NEs)
        print(new_sentences)
        mr_bert_result.append(new_sentences)
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    with open(aeon_prefix + file_name.rstrip("original.json") + "MR_BERT.json", "w", encoding="utf-8") as f:
        json.dump(mr_bert_result, f, ensure_ascii=False)

def run_MR_structure(file_name, aeon_flag, aeon_threshold=0.02):
    # original_results = json.load(open(file_name))[:100]
    original_results = json.load(open(file_name))
    mr_structure = MR_structure()
    mr_structure.MR_init(aeon_flag=aeon_flag, aeon_threshold= aeon_threshold)
    mr_structure_result = []
    for index, sentence in enumerate(original_results):
        # print(index)
        print(str(file_name), index)
        sentence_str, NEs = format_to_string(sentence)
        new_sentences = mr_structure.declarative_2_interrogative(sentence_str, NEs)
        mr_structure_result.append(new_sentences)
    mr_structure.MR_close()
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    with open( aeon_prefix + file_name.rstrip("original.json") + "MR_structure.json", "w", encoding="utf-8") as f:
        json.dump(mr_structure_result, f, ensure_ascii=False)

def run_MR_synonym(file_name, aeon_flag, aeon_threshold=0.01):
    # original_results = json.load(open(file_name))[:100]
    # pdb.set_trace()
    original_results = json.load(open(file_name))
    mr_synonym = MR_synonym()
    mr_synonym.MR_init(aeon_flag=aeon_flag, aeon_threshold= aeon_threshold)
    mr_synonym_result = []
    for index, sentence in enumerate(original_results):
        # print(index)
        print(str(file_name), index)
        # if index == 170:
        #     pdb.set_trace()
        sentence_str, NEs = format_to_string(sentence)
        new_sentences = mr_synonym.generate_syn_sentences(sentence_str, NEs)
        mr_synonym_result.append(new_sentences)
    mr_synonym.MR_close()
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    with open(aeon_prefix + file_name.rstrip("original.json") + "MR_synonym.json", "w", encoding="utf-8") as f:
        json.dump(mr_synonym_result, f, ensure_ascii=False)

def run_MR_intra_shuffle(file_name, aeon_flag, aeon_threshold=0.01):
    # original_results = json.load(open(file_name))[:100]
    original_results = json.load(open(file_name))
    mr_intra_shuffle = MR_intra_shuffle()
    mr_intra_shuffle.MR_init(aeon_flag=aeon_flag, aeon_threshold= aeon_threshold)
    mr_intra_shuffle_result = []
    original_sentences = json.load(open("./Data/bbc.json"))

    for index, sentence in enumerate(original_results):
        # print(index)
        print(str(file_name), index)
        word_list = sentence["sentence"]
        entity_list = sentence["entity"]
        new_sentences = mr_intra_shuffle.intra_sentence_entity_shuffle(word_list, entity_list)
        mr_intra_shuffle_result.append(new_sentences)
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    with open( aeon_prefix + file_name.rstrip("original.json") + "MR_intra_shuffle.json", "w", encoding="utf-8") as f:
        json.dump(mr_intra_shuffle_result, f, ensure_ascii=False)

def run_test_mr_bert(original_file, new_file, api_type = "flair", tag="flair/ner-english-large", aeon_flag=True):
    # pdb.set_trace()
    # original_outputs = json.load(open(original_file))[:100]
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    f=open(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_BERT.out", "w", encoding = 'utf-8')
    original_outputs = json.load(open(original_file))
    new_outputs = json.load(open(new_file))
    if len(original_outputs) != len(new_outputs):
        raise ValueError("The original output does not match the perturbed output")
    if api_type == "flair":
        tagger = SequenceTagger.load(tag)
    apis = APIS()
    sus_list = []
    for i in range(len(original_outputs)):
        print(i)
        sus_info = {}
        suspicious = False
        ori_info_dict = {}
        ori_info_dict["sentence"] = " ".join(original_outputs[i]["sentence"])
        original_prediction = format_to_nameTypeList(original_outputs[i])
        ori_info_dict["entity"] = original_prediction        
        # print(new_outputs[i])
        (changed, current_new_output) = new_outputs[i]
        # print(current_new_output)
        # pdb.set_trace()
        if not changed:
            print("Unchanged:")
            continue
        else:
            print("changed")
        for _current_new_output in current_new_output:
            new_info_dict = {}
            new_sentence = _current_new_output[0]
            new_words = _current_new_output[1]
            if api_type == "flair":
                new_prediction = apis.flair(new_sentence, tagger)
            elif api_type == "google":
                new_prediction = apis.google(new_sentence)
            elif api_type == "azure":
                new_prediction = apis.azure(new_sentence)
            elif api_type == "aws":
                new_prediction = apis.aws(new_sentence)
            elif api_type == "bert_ner":
                new_prediction = apis.bert_ner(new_sentence)
            else:
                raise ValueError("No such API")
            new_info_dict["sentence"] = " ".join(new_prediction["sentence"])
            new_info_dict["entity"] = format_to_nameTypeList(new_prediction)
            # print("original:", format_to_nameTypeList(original_outputs[i]))
            # print("new: ", format_to_nameTypeList(new_prediction))
            # if not check_mr_new_words(original_prediction, format_to_nameTypeList(new_prediction), new_words):
            # pdb.set_trace()
            if check_mr_intersection_new_words(ori_info_dict["sentence"], new_info_dict["sentence"], original_prediction, format_to_nameTypeList(new_prediction)):
                print("suspicious!")
                suspicious = True
                sus_info["index"] = i
                sus_info["original"] = ori_info_dict
                sus_info["new"] = new_info_dict
                sus_list.append(copy.deepcopy(sus_info))                

        print("====================Original====================", file = f)
        print("sentence: ", ori_info_dict['sentence'], file = f)
        print("orientity: ", ori_info_dict['entity'], file = f)
        print("====================Synthesized====================", file = f)
        print("sentence: ", new_info_dict['sentence'] , file = f)
        print("orientity: ", new_info_dict['entity'], file = f)
    print(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_BERT.json")
    with open( aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_BERT.json", "w", encoding="utf-8") as f:
        json.dump(sus_list, f, ensure_ascii=False)

def run_test_mr_structure(original_file, new_file, api_type = "flair", tag = "flair/ner-english-large", aeon_flag=True):
    # original_outputs = json.load(open(original_file))[:100]
    # pdb.set_trace()
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    f=open(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_Structure.out", "w", encoding = 'utf-8')
    original_outputs = json.load(open(original_file))
    new_outputs = json.load(open(new_file))
    if len(original_outputs) != len(new_outputs):
        raise ValueError("The original output does not match the perturbed output")
    apis = APIS()
    if api_type == "flair":
        tagger = SequenceTagger.load(tag)
    sus_list = []
    for i in range(len(original_outputs)):
        print(i)
        sus_info = {}
        ori_info_dict = {}
        ori_info_dict["sentence"] = " ".join(original_outputs[i]["sentence"])
        original_prediction = format_to_nameTypeList(original_outputs[i])
        ori_info_dict["entity"] = original_prediction        
        print("current: ", new_outputs[i])
        (changed, current_new_output) = new_outputs[i]
        if not changed:
            continue
        new_info_dict = {}
        # print(current_new_output)
        new_sentence = current_new_output
        if api_type == "flair":
            new_prediction = apis.flair(new_sentence, tagger)
        elif api_type == "google":
            new_prediction = apis.google(new_sentence)
        elif api_type == "azure":
            new_prediction = apis.azure(new_sentence)
        elif api_type == "aws":
            new_prediction = apis.aws(new_sentence)
        elif api_type == "bert_ner":
            new_prediction = apis.bert_ner(new_sentence)
        else:
            raise ValueError("No such API")
        new_info_dict["sentence"] = " ".join(new_prediction["sentence"])
        new_info_dict["entity"] = format_to_nameTypeList(new_prediction)
        # print("original:", format_to_nameTypeList(original_outputs[i]))
        # print("new: ", format_to_nameTypeList(new_prediction))
        print("check: ", check_mr_equal(original_prediction, format_to_nameTypeList(new_prediction)))
        if not check_mr_equal(original_prediction, format_to_nameTypeList(new_prediction)):
            print("suspicious!")
            suspicious = True
            sus_info["index"] = i
            sus_info["original"] = ori_info_dict
            sus_info["new"] = new_info_dict
            sus_list.append(copy.deepcopy(sus_info))                  

        print("====================Original====================", file = f)
        print("sentence: ", ori_info_dict['sentence'], file = f)
        print("orientity: ", ori_info_dict['entity'], file = f)
        print("====================Synthesized====================", file = f)
        print("sentence: ", new_info_dict['sentence'] , file = f)
        print("orientity: ", new_info_dict['entity'], file = f)
    print(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_structure.json")
    with open( aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_structure.json", "w", encoding="utf-8") as f:
        json.dump(sus_list, f, ensure_ascii=False)


def run_test_mr_synonym(original_file, new_file, api_type = "flair", tag = "flair/ner-english-large", aeon_flag = True):
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    f=open(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_synonym.out", "w", encoding = 'utf-8')
    original_outputs = json.load(open(original_file))
    # original_outputs = json.load(open(original_file))[:100]
    new_outputs = json.load(open(new_file))
    if len(original_outputs) != len(new_outputs):
        raise ValueError("The original output does not match the perturbed output")
    apis = APIS()
    if api_type == "flair":
        tagger = SequenceTagger.load(tag)
    sus_list = []
    for i in range(len(original_outputs)):
        print(i)
        sus_info = {}
        suspicious = False
        ori_info_dict = {}
        ori_info_dict["sentence"] = " ".join(original_outputs[i]["sentence"])
        original_prediction = format_to_nameTypeList(original_outputs[i])
        ori_info_dict["entity"] = original_prediction       
        (changed, current_new_output) = new_outputs[i]
        if not changed:
            continue
        for _current_new_output in current_new_output:
            new_info_dict = {}
            new_sentence = _current_new_output[0]
            new_words = _current_new_output[1]
            if api_type == "flair":
                new_prediction = apis.flair(new_sentence, tagger)
            elif api_type == "google":
                new_prediction = apis.google(new_sentence)
            elif api_type == "azure":
                new_prediction = apis.azure(new_sentence)
            elif api_type == "aws":
                new_prediction = apis.aws(new_sentence)
            elif api_type == "bert_ner":
                new_prediction = apis.bert_ner(new_sentence)
            else:
                raise ValueError("No such API")
            new_info_dict["sentence"] = " ".join(new_prediction["sentence"])
            new_info_dict["entity"] = format_to_nameTypeList(new_prediction)
            if check_mr_intersection_new_words(ori_info_dict["sentence"], new_info_dict["sentence"], original_prediction, format_to_nameTypeList(new_prediction)):
                print("suspicious!")
                # print(new_words)
                # print("new_predition: ", new_prediction)
                suspicious = True
                sus_info["index"] = i
                sus_info["original"] = ori_info_dict
                sus_info["new"] = new_info_dict
                sus_list.append(copy.deepcopy(sus_info))
        
        print("====================Original====================", file = f)
        print("sentence: ", ori_info_dict['sentence'], file = f)
        print("orientity: ", ori_info_dict['entity'], file = f)
        print("====================Synthesized====================", file = f)
        print("sentence: ", new_info_dict['sentence'] , file = f)
        print("orientity: ", new_info_dict['entity'], file = f)
    print(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_synonym.json")  
    with open( aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_synonym.json", "w", encoding="utf-8") as f:
        json.dump(sus_list, f, ensure_ascii=False)
        
def run_test_mr_intra_shuffle(original_file, new_file, api_type = "flair", tag = "flair/ner-english-large", aeon_flag = True):
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
    f=open(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_intra_shuffle.out", "w", encoding = 'utf-8')
    original_outputs = json.load(open(original_file))
    # original_outputs = json.load(open(original_file))[:100]
    new_outputs = json.load(open(new_file))
    if len(original_outputs) != len(new_outputs):
        raise ValueError("The original output does not match the perturbed output")
    apis = APIS()
    if api_type == "flair":
        tagger = SequenceTagger.load(tag)
    sus_list = []
    for i in range(len(original_outputs)):
        print(i)
        sus_info = {}
        suspicious = False
        ori_info_dict = {}
        ori_info_dict["sentence"] = " ".join(original_outputs[i]["sentence"])
        original_prediction = format_to_nameTypeList(original_outputs[i])
        ori_info_dict["entity"] = original_prediction       
        (changed, current_new_output) = new_outputs[i]
        if not changed:
            continue
        for _current_new_output in current_new_output:
            new_info_dict = {}
            new_sentence = _current_new_output
            print(new_sentence)
            if api_type == "flair":
                new_prediction = apis.flair(new_sentence, tagger)
            elif api_type == "google":
                new_prediction = apis.google(new_sentence)
            elif api_type == "azure":
                new_prediction = apis.azure(new_sentence)
            elif api_type == "aws":
                new_prediction = apis.aws(new_sentence)
            elif api_type == "bert_ner":
                new_prediction = apis.bert_ner(new_sentence)
            else:
                raise ValueError("No such API")
            new_info_dict["sentence"] = " ".join(new_prediction["sentence"])
            new_info_dict["entity"] = format_to_nameTypeList(new_prediction)
            # print("original:", format_to_nameTypeList(original_outputs[i]))
            # print("new: ", format_to_nameTypeList(new_prediction))
            if not check_mr_equal(original_prediction, format_to_nameTypeList(new_prediction)):
                print("suspicious!")
                suspicious = True
                sus_info["index"] = i
                sus_info["original"] = ori_info_dict
                sus_info["new"] = new_info_dict
                sus_list.append(copy.deepcopy(sus_info))
    if aeon_flag == True:
        aeon_prefix = "AEON_"
    else:
        aeon_prefix = "NOAEON_"
        
        print("====================Original====================", file = f)
        print("sentence: ", ori_info_dict['sentence'], file = f)
        print("orientity: ", ori_info_dict['entity'], file = f)
        print("====================Synthesized====================", file = f)
        print("sentence: ", new_info_dict['sentence'] , file = f)
        print("orientity: ", new_info_dict['entity'], file = f)
    print(aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_intra_shuffle.json")
    with open( aeon_prefix + "suspicious_" + original_file.rstrip("original.json") + "MR_intra_shuffle.json", "w", encoding="utf-8") as f:
        json.dump(sus_list, f, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process the hyper-parameters")
    parser.add_argument( "--aeon_flag", type = bool, default=True )
    parser.add_argument( "--api_type", type = str, default="flair_conll" )
    parser.add_argument( "--dataset_path", type = str, default="./Data/bbc.json" )
    parser.add_argument( "--down_size", type = bool, default=False )
    args = parser.parse_args()
    print(args.api_type, args.aeon_flag, args.dataset_path, args.down_size)
    aeon_flag = args.aeon_flag
    api_type = args.api_type
    dataset_path = args.dataset_path
    down_size = args.down_size

    if api_type == "flair_conll":
        original_sentences = json.load(open(dataset_path))
        if down_size == True:
            original_sentences = random.Random(14).sample(original_sentences, 100)
        run_flair(original_sentences, tag="flair/ner-english-large", sava_name="bbc_flair_original.json")
        file_name = "bbc_flair_original.json" # original prediction file

        run_MR_BERT(file_name, aeon_flag, aeon_threshold=0.01)
        run_MR_structure(file_name, aeon_flag)
        run_MR_intra_shuffle(file_name, aeon_flag)
        run_MR_synonym(file_name, aeon_flag)
        original_file = file_name

        run_test_mr_bert(original_file, "AEON_bbc_flair_MR_BERT.json", api_type = "flair", tag = "flair/ner-english-large", aeon_flag=True)
        run_test_mr_structure(original_file, "AEON_bbc_flair_MR_structure.json", api_type = "flair", tag = "flair/ner-english-large", aeon_flag=True)
        run_test_mr_intra_shuffle(original_file, "AEON_bbc_flair_MR_intra_shuffle.json", api_type = "flair", tag = "flair/ner-english-large", aeon_flag=True)
        run_test_mr_synonym(original_file, "AEON_bbc_flair_MR_synonym.json", api_type = "flair", tag = "flair/ner-english-large", aeon_flag=True)
    
    elif api_type == "flair_ontonotes":
        original_sentences = json.load(open(dataset_path))
        if down_size == True:
            original_sentences = random.Random(14).sample(original_sentences, 100)
        run_flair(original_sentences, tag="flair/ner-english-ontonotes-large", sava_name="bbc_flair_OntoNotes_original.json")
        file_name = "bbc_flair_OntoNotes_original.json" # original prediction file
        
        run_MR_BERT(file_name, aeon_flag, aeon_threshold=0)
        run_MR_structure(file_name, aeon_flag, aeon_threshold=0)
        run_MR_intra_shuffle(file_name, aeon_flag, aeon_threshold=0)
        run_MR_synonym(file_name, aeon_flag, aeon_threshold=0)
        original_file = file_name

        run_test_mr_bert(original_file, "AEON_bbc_flair_OntoNotes_MR_BERT.json", api_type = "flair", tag="flair/ner-english-ontonotes-large", aeon_flag=True)
        run_test_mr_structure(original_file, "AEON_bbc_flair_OntoNotes_MR_structure.json", api_type = "flair", tag="flair/ner-english-ontonotes-large", aeon_flag=True)
        run_test_mr_intra_shuffle(original_file, "AEON_bbc_flair_OntoNotes_MR_intra_shuffle.json", api_type = "flair", tag="flair/ner-english-ontonotes-large", aeon_flag=True)
        run_test_mr_synonym(original_file, "AEON_bbc_flair_OntoNotes_MR_synonym.json", api_type = "flair", tag="flair/ner-english-ontonotes-large", aeon_flag=True)
    
    elif api_type == "azure":
        original_sentences = json.load(open(dataset_path))
        if down_size == True:
            original_sentences = random.Random(14).sample(original_sentences, 100)
        run_azure(original_sentences)
        file_name = "bbc_azure_original.json" # original prediction file

        run_MR_BERT(file_name, aeon_flag, aeon_threshold=0)
        run_MR_structure(file_name, aeon_flag, aeon_threshold=0)
        run_MR_intra_shuffle(file_name, aeon_flag, aeon_threshold=0)
        run_MR_synonym(file_name, aeon_flag, aeon_threshold=0)
        original_file = file_name

        run_test_mr_bert(original_file, "AEON_bbc_azure_MR_BERT.json", api_type = "azure", aeon_flag=True)
        run_test_mr_structure(original_file, "AEON_bbc_azure_MR_structure.json", api_type = "azure", aeon_flag=True)
        run_test_mr_intra_shuffle(original_file, "AEON_bbc_azure_MR_intra_shuffle.json", api_type = "azure", aeon_flag=True)
        run_test_mr_synonym(original_file, "AEON_bbc_azure_MR_synonym.json", api_type = "azure", aeon_flag=True)
    
    elif api_type == "aws":
        original_sentences = json.load(open(dataset_path))
        if down_size == True:
            original_sentences = random.Random(14).sample(original_sentences, 100)
        run_aws(original_sentences)
        file_name = "bbc_aws_original.json" # original prediction file

        run_MR_BERT(file_name, aeon_flag, aeon_threshold=0)
        run_MR_structure(file_name, aeon_flag, aeon_threshold=0)
        run_MR_intra_shuffle(file_name, aeon_flag, aeon_threshold=0)
        run_MR_synonym(file_name, aeon_flag, aeon_threshold=0)
        original_file = file_name

        run_test_mr_bert(original_file, "AEON_bbc_aws_MR_BERT.json", api_type = "aws", aeon_flag=True)
        run_test_mr_structure(original_file, "AEON_bbc_aws_MR_structure.json", api_type = "aws", aeon_flag=True)
        run_test_mr_intra_shuffle(original_file, "AEON_bbc_aws_MR_intra_shuffle.json", api_type = "aws", aeon_flag=True)
        run_test_mr_synonym(original_file, "AEON_bbc_aws_MR_synonym.json", api_type = "aws", aeon_flag=True)