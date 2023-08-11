import json
import os

class Transer:
    # txt file to json file (pre-process the sentences)
    def txt_preprocess(self, file_path, output_path):
        result = []
        with open(file_path, "r", encoding = 'utf-8') as f:
            for line in f:
                if "-" in line:
                    continue
                # deal with separating signs
                line = line.strip('\n')
                line = line.replace('\0', ' ')
                line = line.replace("\"", ' ')
                # deal with "'s" issues
                line = line.replace("'s", " 's")
                line = line.replace("  's", " 's")
                # deal with "'m"
                line = line.replace("'m", " 'm")
                line = line.replace("  'm", " 'm")
                # deal with "'re"
                line = line.replace("'re", " 're")
                line = line.replace("  're", " 're")
                # deal with "()" issues
                line = line.replace("(", "( ")
                line = line.replace(")", " )")
                line = line.replace("(  ", "( ")
                line = line.replace("  )", " )")
                # deal with "n't"
                line  = line.replace("n't" , " n't")
                line  = line.replace("  n't" , " n't")
                sentence = []
                text = ""
                for index, c in enumerate(line):
                    if index > 0 and index < len(line)-1 and c == "'": 
                        if line[index-1] != " " and line[index+1] == " ":
                            sentence.append(text)
                            text = ""
                            sentence.append(c)
                        else:
                            text = text + c
                    elif c == " " or c == "." or c == '"' or c == "," or c == ".":
                        if(text != ""): sentence.append(text)
                        text = ""
                        if(c != " "): sentence.append(c)
                    else: text = text + c
                if(text != ""): sentence.append(text)
                result.append(" ".join(sentence))
        print(len(result))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            
if __name__ == "__main__":
    transer = Transer()
    for file in os.listdir('.'):
        print(file)
        if file.endswith(".txt"): transer.txt_preprocess(file_path=file, output_path=file[:-4]+".json")
    