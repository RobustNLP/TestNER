import sys
# import the AEON, please change the follosing path with the path ofaeon.py
# sys.path.insert(1, '$HOME$/NER_MRs/MRs/AEON/utils')
from aeon import *
import itertools
import random
import nltk
import ipdb

class MR_intra_shuffle:
    def MR_init(self, aeon_flag = True, evaluate = False, aeon_threshold = 0.01):
        self.scorer_default = Scorer(
            8, 
            'bert-base-uncased', 
            'princeton-nlp/sup-simcse-bert-base-uncased',
            True
        )
        self.permutation_bound = 4 # max number of entities to generate permutations
        self.generate_bound = 8 # max sentences to be generated repecting one type (before filtering)
        self.random_seed = 0
        self.max_sentences = 5
        # self.AEON_threshold = 0
        self.AEON_threshold = aeon_threshold
        self.filter_flag = aeon_flag
        self.evaluate = evaluate
    
    def __get_AEON_score(self, sentence):
        return self.scorer_default.compute_naturalness(sentence)
    
    def intra_sentence_entity_shuffle(self, word_list, entity_list):
        frame_word_list = word_list.copy() # a frame word list for generating new sentences
        entity_type_pool = {} # a pool to store all the entities of the same type
        
        # get the entity type pool and replace original entities with place hoders
        for (entity_name, entity_type, entity_indexes) in entity_list:
            current_type = entity_type
            if current_type not in entity_type_pool:
                entity_type_pool[current_type] = []
            entity_type_pool[current_type].append(entity_name)
            for index in entity_indexes:
                type_hoder = "<entity_type>" + current_type + "+" + entity_name
                frame_word_list[index] = type_hoder
        
        # concate the adjacent place holders with the same type
        # the new_word_list is a new list of words after concating the place holders
        last_word = None
        new_word_list = []
        if len(frame_word_list) < 2:
            return (False, [])
        for i in range(0, len(frame_word_list)):
            curr = frame_word_list[i]
            if curr.startswith("<entity_type>"):
                if last_word == curr:
                    continue # do not append to the new word list
                else:
                    last_word = curr
                    new_word_list.append(curr)
            else:
                last_word = curr
                new_word_list.append(curr)
        
        # generate permuatations or shuffled lists of entity names
        entity_types = list(entity_type_pool.keys())
        new_entity_pool_list = {}
        for e_type in entity_types:
            new_entity_pool_list[e_type] = []
            new_type_pool_list = []
            if len(entity_type_pool[e_type]) <= self.permutation_bound: # get permutation if there are not many entities
                new_type_pool_list = list(itertools.permutations(entity_type_pool[e_type]))
            else: # otherwise randomly generate
                seed_now = self.random_seed
                for i in range(0, self.generate_bound):
                    temp_list = entity_type_pool[e_type]
                    random.Random(seed_now).shuffle(temp_list)
                    new_type_pool_list.append(temp_list)
                    seed_now += 1
                    
            new_entity_pool_list[e_type] = new_type_pool_list
        
        # generate new sentences
        ori_number = 0
        sim_number = 0
        aeon_number = 0
        new_sentences = []
        new_sentences.append(new_word_list)
        occurences_all = []
        # ipdb.set_trace()
        for e_type in entity_types:
            i = 0
            occurences = [i for i, x in enumerate(new_word_list) if x.startswith("<entity_type>"+e_type)]
            occurences_all.extend(occurences)
            temp_sentences = []
            while i < len(new_sentences):
                sentence_list = new_sentences[i]
                for entity_names in new_entity_pool_list[e_type]:
                    if len(entity_names) < len(occurences):
                        continue
                    new_sentence_list = sentence_list.copy()
                    index = 0
                    while index < len(occurences):
                        new_sentence_list[occurences[index]] = entity_names[index]
                        index += 1
                    temp_sentences.append(new_sentence_list)
                i += 1
            new_sentences.clear()
            for item in temp_sentences:
                new_sentences.append(item)
        
        candiate_sentences = new_sentences
        ori_number = len(candiate_sentences)
        sim_number = len(candiate_sentences)
        
        # AEON filter
        result_sentences = []
        original_sentence = " ".join(word_list)
        if self.filter_flag:
            ori_score = self.__get_AEON_score(original_sentence)
            for sentence_list in candiate_sentences:
                new_sentence = " ".join(sentence_list)
                if original_sentence == new_sentence:
                    continue
                new_score = self.__get_AEON_score(new_sentence)
                if new_score < ori_score - self.AEON_threshold:
                    continue
                result_sentences.append(new_sentence)
        else:
            result_sentences = candiate_sentences
        aeon_number = len(result_sentences)
        
        if len(result_sentences) == 0:
            return (False, [])
        if self.evaluate == True:
            return [ori_number, sim_number, aeon_number]
        return (True, result_sentences)
        
if __name__ == "__main__":
    mr_intra_shuffle = MR_intra_shuffle()
    mr_intra_shuffle.MR_init()
    ent_info = {"sentence": ["On", "a", "video", "posted", "on", "the", "platform", "Rumble", ",", "Jones", "said", "he", "did", "not", "care", "about", "being", "on", "Twitter", ",", "reports", "news", "website", "Axios", "."], "entity": [["Rumble", "PRODUCT", [7]], ["Jones", "PERSON", [9]], ["Twitter", "PRODUCT", [18]], ["Axios", "ORG", [23]]]}
    print(mr_intra_shuffle.intra_sentence_entity_shuffle(ent_info["sentence"], ent_info["entity"]))

    