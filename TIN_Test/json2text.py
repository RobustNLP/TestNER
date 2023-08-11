import json
import os
import pdb


for file in os.listdir('.'):
    temp_susp = []
    temp_list = []
    if file[-5:] == '.json' and  (  ("AEON_suspicious" or "NOAEON_suspicious") in file  ):
        with open(file, 'r' ) as f:
            temp_susp = json.load(f)
            wf = open(file[:-5] + '.txt', "w", encoding='UTF-8')
            for idx, ele in enumerate(temp_susp):
                wf.write("====================Suspicious Issues====================" + "\n")
                wf.write( "original: " + ele['original']['sentence'] +"\n")
                wf.write( "entities: " + str(ele['original']['entity']) +"\n")
                wf.write( "synthetic: " +  ele['new']['sentence'] + "\n" )
                wf.write( "entities " + str(ele['new']['entity']) +"\n" )
                wf.write("=================================================" + "\n")