## Reparing the raised suspicious issues

### Environment

```
python == 3.7
pytorch == 1.7.1
Transformers == 3.3.0
```

### Run

To run the repairing part of **TIN**, you need import *Repairer* from *repair.py* in your code first.

 Then you can execute the following python codes to automatically repair the NER issues raised by the testing part:

```python
MY_REPAIRER = Repairer() 
MY_REPAIRER.bert_init() # Initialize the BERT
MY_REPAIRER.repair_suspicious_issue(input_file, output_file, api)
```

 , where the *input_file* represents the path for the json file of the suspicious issues,  the *output_file* represents the path of the json file to store the NER repairing result, and *api* is the function to obtain the NER prediction from the NER systems.

We provide four api functions including *flair*, *flair_OntoNotes*, *azure* and *aws*, you can import them from the *apis.py*. In practice, you can invoke the *flair* and *flar_OntoNotes* directly, and invoke the *azure* and *aws* after you fill your keys in the *apis.py*. It is also availiable that you write own api function to repair other NER system, but you need ensure the output format of your api function is consistent with the api functions provided.

For example, you can run these codes in the repair.py directly and the program will show the repairing result in the ```data/suspicious_flair_repair.json```

```python
apis = APIS()
NER_repairer = Repairer()
NER_repairer.bert_init()
NER_repairer.repair_suspicious_issue("./data/suspicious_flair.json",
                        "./data/suspicious_flair_repair_json", apis.flair)
```

We provide a demo of repairing, you can directly run:
```
python repair.py
```
## Example results
We provide the example results including transformation results and suspicious issues, the example results are in the `{ROOT_DIR}/repair_evaluation_result` folder.