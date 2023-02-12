import os

def generate_fasttext_train_data(train_set):
    with open('data/out/spam-train.txt', 'w', encoding="utf-8") as f:
        for idx, row in train_set.iterrows():
            f.write("__label__" + row.label + " " + row.sentence + "\n")