from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import *
from nltk.stem.wordnet import WordNetLemmatizer
import nltk.data
import stanza
import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'
import sys
import ipdb
# import the AEON, please change the follosing path with the path ofaeon.py
# sys.path.insert(1, '$HOME$/NER_MRs/MRs/AEON/utils')
from aeon import *
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

class MR_structure:
    def MR_init(self, aeon_flag, evaluate = False, aeon_threshold = 0.02):
        self.nlp_pos = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
        self.nlp = StanfordCoreNLP(
            str(pathlib.Path().absolute())+'/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
        # self.nlp = StanfordCoreNLP(
        #     r'$HOME$/NER_MRs/corenlp/stanford-corenlp-latest/stanford-corenlp-4.4.0')
        self.scorer_default = Scorer(
            8, 
            'bert-base-uncased', 
            'princeton-nlp/sup-simcse-bert-base-uncased',
            True
        )
        self.AEON_threshold = aeon_threshold
        self.filter_flag = aeon_flag
        self.evaluate = evaluate
    
    def __get_AEON_score(self, sentence):
        return self.scorer_default.compute_naturalness(sentence)
        
    def MR_close(self):
        self.nlp.close()
    
    def __get_entity_ids(self, NEs, word_list):
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
    
    def __checkFirstNoun(self, Snode, NEs):
        # first convert the subtree to a sentence
        tokens = Snode.leaves()
        for i in range(0, len(tokens)):
            if tokens[i] == "-LRB-":
                tokens[i] = "("
            elif tokens[i] == "-RRB-":
                tokens[i] = ")"
        S_sentence = " ".join(tokens)
        entity_ids = self.__get_entity_ids(NEs, tokens)
        # check if "I" is the first word
        if len(S_sentence) == 0:
            return False
        if tokens[0].casefold() == "i":
            return True
        # check if the first word is in the entity words
        if 0 in entity_ids:
            return True
        # check pos tag: if NNP
        doc = self.nlp_pos(S_sentence)
        word = doc.sentences[0].words[0]
        if word.xpos == "NNP" or word.xpos == "NNPS":
            return True
        return False
    
    def __get_leftest_leaf(self, node):
        temp = node
        while type(temp) != str:
            temp = temp[0]
        return temp

    def __get_rightest_leaf(self, node):
        temp = node
        while type(temp) != str:
            temp = temp[len(temp)-1]
        return temp

    def __get_leftest_label(self, node):
        temp = node
        ref_list = []
        while type(temp) != str and type(temp[0]) != str:
            temp = temp[0]
            ref_list.append(0)
        return temp, ref_list
    
    def __find_sentence_nodes(self, root):
        node_list = []
        nodeIndex_list = []

        def find_node(node):
            find_list = []
            find_list.append((node, [])) # [0]-> record the indexing info of the current node
            while len(find_list) > 0:
                curr = find_list.pop(0)
                if type(curr[0]) == str: # leaf
                    continue
                else:
                    if curr[0].label() == "SQ" or curr[0].label() == "SBAR":
                        continue
                    if curr[0].label() == "S":
                        node_list.append(curr[0])
                        nodeIndex_list.append(curr[1])
                        continue
                    i = 0
                    for child in curr[0]:
                        temp_list = curr[1].copy()
                        temp_list.append(i)
                        find_list.append((child, temp_list))
                        i += 1
        
        find_node(root)
        return node_list, nodeIndex_list
    
    def __declarative_2_interrogative_Tree(self, sentenceNode, NEs, mode):
    
        # change capital letter of the NP in the sentence
        def changeCapLetter_lower(node):
            nn_node, ref_list = self.__get_leftest_label(node)
            label_nn = nn_node.label()
            string_nn = nn_node[0]
            if (string_nn.isupper() and len(string_nn)>2) or self.__checkFirstNoun(node, NEs):
                   return node
            # else change the noun to the lower case format
            var_string = "node"
            assign_string = "=string_nn.lower()"
            for i in ref_list:
                var_string += "[0]"
            var_string += "[0]"
            exec(var_string + assign_string)
            return node

        # change the first letter of the sentence to upper case
        def changeCapLetter_upper(node):
            nn_node, ref_list = self.__get_leftest_label(node)
            label_nn = nn_node.label()
            string_nn = nn_node[0]
            # else change the noun to the lower case format
            string_nn = string_nn[0].upper() + string_nn[1:]
            var_string = "node"
            assign_string = "=string_nn"
            for i in ref_list:
                var_string += "[0]"
            var_string += "[0]"
            exec(var_string + assign_string)
            return node
        
        # change the declarative sentence to a general question
        def d2q_GEN(Snode, isSentence):
            # if the first word of the sentence is not a word
            if self.__get_leftest_leaf(Snode).isalnum() == False:
                return Snode, False
            # find the VP in S, S should be NP + VP
            if type(Snode) == str:
                return Snode, False
            if Snode[0].label() != "NP":
                return Snode, False
            curr_node = Snode[0]
            i = 0
            for child in Snode:
                if child.label() == "VP":
                    curr_node = child
                    break
                i += 1
            if i == 0 or i == len(Snode):
                return Snode, False # no VP
            # check the VB form of the VP
            # modal verb
            if type(curr_node[0]) == str: # not supported format
                return Snode, False
            if len(Snode) <= 1:
                # in case the sentence only consist a NP and VP, this does not follow the rule
                return Snode, False
            if curr_node[0].label() == "MD":
                # modal verb
                Snode = changeCapLetter_lower(Snode)
                temp = curr_node.pop(0)
                Snode.insert(0, Tree("MD", temp))
                if Snode[len(Snode)-1].label() == ".":
                    Snode[len(Snode)-1] = Tree(".", ["?"])
                if isSentence: Snode = changeCapLetter_upper(Snode)
                return Snode, True
            elif curr_node[0].label().startswith("VB"):
                if type(curr_node[0][0]) != str or len(curr_node)<2:
                    return Snode, False
                # be verbs (eg: is doing)
                elif curr_node[0][0] == "am" or curr_node[0][0] == "is" or curr_node[0][0] == "is" or curr_node[0][0] == "was" or curr_node[0][0] == "are" or curr_node[0][0] == "were":
                    # swqp the NP and the VB
                    Snode = changeCapLetter_lower(Snode)
                    temp = curr_node.pop(0)
                    Snode.insert(0, Tree("VB", temp))
                    if Snode[len(Snode)-1].label() == ".":
                        Snode[len(Snode)-1] = Tree(".", ["?"])
                    if isSentence: Snode = changeCapLetter_upper(Snode)
                    return Snode, True
                # participated past verbs (eg: have done)
                # check support format
                elif type(curr_node[1]) == str:
                    return Snode, False
                elif type(curr_node[1][0]) == str:
                    return Snode, False
                elif (curr_node[0][0] == "have" or curr_node[0][0] == "has" or curr_node[0][0] == "had")\
                    and curr_node[1][0].label() == "VBN":
                    # swqp the NP and the VB (have)
                    Snode = changeCapLetter_lower(Snode)
                    temp = curr_node.pop(0)
                    Snode.insert(0, Tree("VB", temp))
                    if Snode[len(Snode)-1].label() == ".":
                        Snode[len(Snode)-1] = Tree(".", ["?"])
                    if isSentence: Snode = changeCapLetter_upper(Snode)
                    return Snode, True
                # normal verbs (do)
                else:
                    if curr_node[0].label() == "VB" or curr_node[0].label() == "VBG" or curr_node[0].label() == "VBP":
                        if type(curr_node[0][0]) != str: return Snode
                        Snode = changeCapLetter_lower(Snode)
                        word = curr_node[0][0]
                        word = WordNetLemmatizer().lemmatize(word,'v')
                        curr_node[0][0] = word
                        if isSentence: Snode.insert(0, Tree("VB", ["Did"])) # "do" may cause some problems
                        else: isSentence: Snode.insert(0, Tree("VB", ["did"]))
                        if Snode[len(Snode)-1].label() == ".":
                            Snode[len(Snode)-1] = Tree(".", ["?"])
                        return Snode, True
                    elif curr_node[0].label() == "VBZ":
                        if type(curr_node[0][0]) != str: return Snode
                        Snode = changeCapLetter_lower(Snode)
                        word = curr_node[0][0]
                        word = WordNetLemmatizer().lemmatize(word,'v')
                        curr_node[0][0] = word
                        if isSentence: Snode.insert(0, Tree("VB", ["Does"]))
                        else: Snode.insert(0, Tree("VB", ["does"]))
                        if Snode[len(Snode)-1].label() == ".":
                            Snode[len(Snode)-1] = Tree(".", ["?"])
                        return Snode, True
                    elif curr_node[0].label() == "VBD" or curr_node[0].label() == "VBN":
                        if type(curr_node[0][0]) != str: return Snode
                        Snode = changeCapLetter_lower(Snode)
                        word = curr_node[0][0]
                        word = WordNetLemmatizer().lemmatize(word,'v')
                        curr_node[0][0] = word
                        if isSentence: Snode.insert(0, Tree("VB", ["Did"]))
                        else: Snode.insert(0, Tree("VB", ["did"]))
                        if Snode[len(Snode)-1].label() == ".":
                            Snode[len(Snode)-1] = Tree(".", ["?"])
                        return Snode, True
                    else:
                        return Snode, False
            else:
                return Snode, False
        
        shouldChange = False # when there is one change, this is set to true
        Snode_list, SnodeIndex_list = self.__find_sentence_nodes(sentenceNode)
        for i in range(0, len(Snode_list)):
            if mode == "GEN":
                isSentence = True # check if the current node represent a complete sentence
                listName = "sentenceNode"
                assignName = "=Snode"
                checkPrevName = "sentenceNode"
                for k in range(0, len(SnodeIndex_list[i])):
                    # change the subtree
                    listName += "[" + str(SnodeIndex_list[i][k]) + "]"
                    if k == len(SnodeIndex_list[i])-1:
                        if SnodeIndex_list[i][k] > 0:
                            checkPrevName += "[" + str(SnodeIndex_list[i][k]-1) + "]"
                        else:
                            checkPrevName = ""
                    else:
                        checkPrevName += "[" + str(SnodeIndex_list[i][k]) + "]"
                try:
                    prevWord = self.__get_rightest_leaf(eval(checkPrevName))
                    if prevWord == ",":
                        isSentence = False
                    else:
                        isSentence = True
                except:
                    isSentence = True
                Snode, change = d2q_GEN(Snode_list[i], isSentence)
                if change == False: continue
                shouldChange = True
                if type(eval(listName)) == str:
                    return
                exec(listName+assignName)
        return shouldChange
    
    def __declarative_2_interrogative_single(self, sentence, NEs, mode = "GEN"):
        changed = False
        cap = sentence.isupper()
        try:
            parsingTree = Tree.fromstring(self.nlp.parse(sentence))
        except:
            print("something wrong with the stanford parser: error creating parsing tree")
            print("issue sentence: ", sentence)
            return False, sentence
        temp = self.__declarative_2_interrogative_Tree(parsingTree, NEs, mode)
        if temp == False:
            # print("not changed !")
            return False, sentence
        else:
            changed = True
            tokens = parsingTree.leaves()
            for i in range(0, len(tokens)):
                if tokens[i] == "-LRB-":
                    tokens[i] = "("
                elif tokens[i] == "-RRB-":
                    tokens[i] = ")"
                if cap and tokens[i].isupper() == False:
                    tokens[i] = tokens[i].upper()
            return_string = " ".join(tokens)
            if " - " not in sentence: return_string = return_string.replace(" - ", "-")
            if "?" not in return_string: 
                if return_string[len(return_string)-1] == ".":
                    return_string = return_string[:len(return_string)-1] + "?"
                else:
                    return_string += "?" # add question mark at the last if there is no punct
            return changed , return_string
    
    def declarative_2_interrogative(self, sentence, NEs):
        ori_number = 0
        sim_number = 0
        aeon_number = 0
        multi_changed = False
        if len(sentence) == 0:
            if self.evaluate == True:
                return [ori_number, sim_number, aeon_number]
            return (multi_changed, sentence) # empty sentence
        sentence_list = split_into_sentences(sentence)
        new_sentence_list = []
        for s in sentence_list:
            changed, single_sentence = self.__declarative_2_interrogative_single(s, NEs)
            if changed:
                multi_changed = True
                ori_number = 1
                sim_number = 1
            new_sentence_list.append(single_sentence)
        if not multi_changed:
            if self.evaluate == True:
                return [ori_number, sim_number, aeon_number]
            return (multi_changed, sentence)
        new_sentence = " ".join(new_sentence_list)
        print(new_sentence)
        # use aeon to filter
        if self.filter_flag:
            ori_score = self.__get_AEON_score(sentence)
            new_score = self.__get_AEON_score(new_sentence)
            if new_score < ori_score - self.AEON_threshold:
                multi_changed = False
                new_sentence = sentence
            else:
                aeon_number = 1
        if self.evaluate == True:
            print(self.evaluate)
            return [ori_number, sim_number, aeon_number]
        return (multi_changed, new_sentence)

       
    def show(self, sentence):
        parsingTree = Tree.fromstring(self.nlp.parse(sentence))
        print(self.nlp.parse(sentence))
        parsingTree.pretty_print()
        
if __name__ == "__main__":
    mr_structure = MR_structure()
    mr_structure.MR_init(aeon_flag=True)
    mr_structure.declarative_2_interrogative("I am good .", [])
    mr_structure.MR_close()