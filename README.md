# Weakly Supervised Spam-Label Classification
DSC180 Quarter 2 Capstone Project 

Using a list of categories and words that represent these categories, we classify harmful spam messages into categories such as insurance scams, medical sales, software sales, and more. Doing so, we hope to alleviate the burden on non technical people in todays world as spammers continue to get by detection systems - we want to find and highlight a pattern throughout them all. Leveraging models ranging from simple methods like TFIDF to complex large language models such as ConWea with BERT, we examine the differences between these models and if it is worth using such big, computation costly models.

You can see more details about our project on our [website](https://gbirch11.github.io/SpamLabelClassifier/).

# Data
The data is available on [Google Drive](https://drive.google.com/drive/folders/1uTRzRPkom6nUtRB2D4pOi8uOpSpqst7m?usp=share_link)\
Please unzip and place the files into the following locations; \
Annotated Spam Messages -> ```data/raw/spam/Annotated/``` \
Unannotated Spam Messages -> ```data/raw/spam/Unannotated/``` \
Non-spam (Ham) Messages -> ```data/raw/ham/``` 


The dataset should contain the following files:
1) Annotated Spam Messages \
  ex) ```data/raw/spam/Annotated/medical-sales/xyz.txt```
    * Where xyz is any file name that was annotated to be medical sale spam
    * Other folders follow same pattern for each category
2) Non-spam (ham) Messages \
  ex) ```data/raw/ham/xyz.txt```
3) Seedwords JSON file \
  ex) ```data/out/seedwords.json```

## Running the Project
**DSMLP Command**
``` 
launch.sh -i gbirch11/dsc180b [-m d] [-g 1]
```
Note: -m is an optional argument to include more RAM on the machine; HIGLHLY RECOMMEND setting $d$ to 16 or 32 for faster processing \
Also highly recommended to run with -g 1, especially if running ConWea model.
``` 
launch.sh -i gbirch11/dsc180b -m 32 -g 1
```
<br> <br>
To run this project, execute the following command;
```
python run.py [test | data]
```
Note: If running ```python run.py test``` \
Very simple set of test data will be used to produce results. \\
Result trend not consistent with running on full dataset.

If running ```python run.py data```: \
Whole dataset will be used to produce results.

Example commands include: \
``` python run.py test ``` \
``` python run.py data ```

Note: The above commands only run on the TF-IDF, Word2Vec, and FastText models. To run our best model, ConWea, see the section below.

## Running the ConWea Model
Since ConWea is a huge model using BERT, we have separated this model into the following separate commands;
1) Navigate to the ConWea model directory using \
``` cd src/models/ConWea ``` <br> <br>
2) To contextualize the corpus and seed words run \
a) For testing: ``` python contextualize.py --dataset_path "../../../test/testdata/" --temp_dir "temp/" --gpu_id 0 ``` \
b) For full data: ``` python contextualize.py --dataset_path "../../../data/raw/spam/Annotated/" --temp_dir "temp/" --gpu_id 0 ```  <br> <br>
3) To train model + observe results run \
a) For testing: ``` python train.py --dataset_path "../../../test/testdata/" --gpu_id 0 ``` \
b) For full data: ``` python train.py --dataset_path "../../../data/raw/spam/Annotated/" --gpu_id 0 ```  <br> <br>

Note: Be warned that running ConWea on the full dataset will ~ 3 hours to run. Running ConWea on test data runs in ~ 20 minutes. <br>
Note: ConWea trains using multiple layers and tons of epochs, since our test data is small it is safe to interrupt the terminal (CTRL+C) after first iteration has occured. The layers are kept for consistency for full datasets.
