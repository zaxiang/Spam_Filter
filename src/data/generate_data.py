import os
import pandas as pd
from src.features.features import *


def read_data(target):
    labels = ["insurance-etc","investment", "medical-sales", "phising", "sexual", "software-sales"]
    text = []
    classes = []
    for cla in labels:
        path = "data/raw/spam/Annotated/"
        if target == 'test':
            path = "test/testdata/"
        all_files = os.listdir(path + cla)
        for fil in all_files:
            if fil.endswith(".txt"):
                file_path = path + cla + "/" + fil
                with open(file_path, 'r', encoding='ISO-8859-1') as f:
                    text.append(cleaning(str(f.read())))
                    classes.append(cla)
    return pd.DataFrame({'sentence':text, 'label':classes})



def generate_fasttext_train_data(train_set):
    with open('data/out/spam-train.txt', 'w', encoding="utf-8") as f:
        for idx, row in train_set.iterrows():
            f.write("__label__" + row.label + " " + row.sentence + "\n")