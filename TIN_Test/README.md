# TIN

A toolkit for testing and repairing NER systems



## Environment requirements

* nltk
* stanfordcorenlp
* transformers

## Related packages

We use some packages to implement the toolkit:

* Stanford CoreNLP (https://stanfordnlp.github.io/CoreNLP/)
* flair (https://github.com/flairNLP/flair)
* Microsoft Azure: Named Entity Recognition (https://learn.microsoft.com/en-us/azure/cognitive-services/language-service/named-entity-recognition/overview)
* Amazon Segamaker: Named Entity Recognition (https://docs.aws.amazon.com/sagemaker/latest/dg/sms-named-entity-recg.html)
* Sense2Vec (https://github.com/explosion/sense2vec)
* AEON (https://github.com/CUHK-ARISE/AEON)

Specifically, you need to download some of the packages to the `{ROOT_DIR}/Data` directory, the packages includes Stanford CoreNLP and Sense2Vec.

For corenlp, you need to download corenlp (https://stanfordnlp.github.io/CoreNLP/) add the whole `corenlp` folder to the `{ROOT_DIR}`.

For Sense2vec, you need to download the s2v_reddit_2015_md package (s2v_reddit_2015_md) and add the `s2v_old` folder to `{ROOT_DIR}`.

The final `{ROOT_DIR}` directory contains:
```shell
AEON
apis.py
corenlp
Data
Example_results
json2text.py
mr_perturb.py
__pycache__
s2v_old
Transformations
```

## Work Pipeline Example

(Optional) Before you run **Tin**, you should preprocess your dataset. You need to put your dataset in the `${REPO_ROOT_DIR}/Data` directory. Then you need to run `transer.py` to preprocess the data. Note: your dataset should be in txt format where each line contains one sentence of the string format.

```
cd ${REPO_ROOT_DIR}/Data
python transer.py
```

**Tin** includes 2 parts: 1. the testing part, 2. the repair part. You should first run the testing part, then you can run the repair part based on the result of the testing part. The following part demonstrates a typical work flow

1. Testing: generate suspicious issues

   ```shell
   cd ${REPO_ROOT_DIR}
   python mr_perturb.py
   python json2text.py # convert json to txt to check the suspicious issues
   ```

2. Repair: locate suspicious entities and repair



## Testing and generate suspicious issues

To run the testing step, you should run the `mr_perturb.py` in the root directory. You can adjust the following arguments to run different settings:

* aeon_flag (bool): set whether to use naturalness filter

  > **True**: use naturalness filter
  >
  > **False**: do not use naturalness filter 
  >
  > By default aeon_flag is set to **True**

* api_type (string): set which api or model to test

  > Specifically, you can use the following api types:
  >
  > **flair_conll**: test on flair's pretrained model of conll03 dataset
  >
  > **flair_ontonotes**: test on flair's pretrained model of ontonotes5 dataset
  >
  > **azure**: test on Microsoft Azure NER api
  >
  > **aws**: test on AWS Segamaker NER api
  >
  > By default the api_type is set to **flair_conll**

* dataset_path (string): set the path of your dataset

  > Set the path of your data (which is in the json format)
  >
  > By default, **dataset_path** = ${REPO_ROOT_DIR}/Data/bbc.json

* down_size (bool): set whether you only want to test 100 cases

  > When the dataset can be large, the program runs for a long time.
  >
  > You can set down_size to **True** if you want to run on only 100 cases

One example of running `mr_perturb.py`

```shell
cd ${REPO_ROOT_DIR}
python mr_perturb.py --aeon_flag=True --api_type=azure --dataset_path=./Data/bbc.json --down_size=False
python json2text.py
```

## Example results
We provide the example results including transformation results and suspicious issues, the example results are in the `{ROOT_DIR}/Example_results` folder.