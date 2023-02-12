# Spam_Filter
DSC180 Quarter 2 Capstone Project \
Using a list of categories and words that represent these categories, we classify harmful spam messages into categories such as insurance scams, medical sales, software sales, and more. Doing so, we hope to alleviate the burden on non technical people in todays world as spammers continue to get by detection systems - we want to find and highlight a pattern throughout them all. Leveraging models ranging from simple methods like TFIDF to complex large language models such as BERT and ConWea, we examine the differences between these models and if it is worth using such big, computation costly models.

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
