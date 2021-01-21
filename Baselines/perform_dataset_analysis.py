import json
from nltk.tokenize import word_tokenize
import nltk
import numpy as np

"""
DESCRIPTION:
This file perform analysis on the language contained within different Embodied Vision and Language datasets.
"""


def do_way_data(fileName):
    data = json.load(open(way_data + fileName))
    for data_obj in data:
        text = data_obj["dialog"]
        words = word_tokenize(text.lower())
        for word in words:
            vocab.add(word)
        nouns = 0
        adjs = 0
        preps = 0
        for t in nltk.pos_tag(words):
            pos = t[1]
            if pos in ["NN", "NNS", "NNP", "NNPS"]:
                nouns += 1
            elif pos in ["JJ", "JJR", "JJS"]:
                adjs += 1
            elif pos in ["IN", "TO"]:
                preps += 1
        avg_nouns.append(nouns)
        avg_adjs.append(adjs)
        avg_preps.append(preps)
    print("vocab size:", len(vocab))
    print("# nouns:", np.mean(avg_nouns))
    print("# adj:", np.mean(avg_adjs))
    print("# prep:", np.mean(avg_preps))


def do_vln_data(fileName):
    data = json.load(open(vln_data + fileName))
    for data_obj in data:
        for text in data_obj["instructions"]:
            words = word_tokenize(text.lower())
            t_length.append(len(words))
            for word in words:
                vocab.add(word)
            nouns = 0
            adjs = 0
            preps = 0
            for t in nltk.pos_tag(words):
                pos = t[1]
                if pos in ["NN", "NNS", "NNP", "NNPS"]:
                    nouns += 1
                elif pos in ["JJ", "JJR", "JJS"]:
                    adjs += 1
                elif pos in ["IN", "TO"]:
                    preps += 1
            avg_nouns.append(nouns)
            avg_adjs.append(adjs)
            avg_preps.append(preps)
    print("vocab size:", len(vocab))
    print("# nouns:", np.mean(avg_nouns))
    print("# adj:", np.mean(avg_adjs))
    print("# prep:", np.mean(avg_preps))


def do_cvdn_data(fileName):
    data = json.load(open(cvdn_data + fileName))
    for data_obj in data:
        dh = data_obj["dialog_history"]
        text = ""
        for m in dh:
            text += m["message"] + " "
        text = text.strip()
        words = word_tokenize(text.lower())
        t_length.append(len(words))
        for word in words:
            vocab.add(word)
        nouns = 0
        adjs = 0
        preps = 0
        for t in nltk.pos_tag(words):
            pos = t[1]
            if pos in ["NN", "NNS", "NNP", "NNPS"]:
                nouns += 1
            elif pos in ["JJ", "JJR", "JJS"]:
                adjs += 1
            elif pos in ["IN", "TO"]:
                preps += 1
        avg_nouns.append(nouns)
        avg_adjs.append(adjs)
        avg_preps.append(preps)
    print("vocab size:", len(vocab))
    print("# nouns:", np.mean(avg_nouns))
    print("# adj:", np.mean(avg_adjs))
    print("# prep:", np.mean(avg_preps))


def do_ttw_data(fileName):
    data = json.load(open(ttw_data + fileName))
    print(len(data))
    for data_obj in data:
        text = ""
        for i in data_obj["dialog"]:
            m = i["text"]
            if m == "ACTION:TURNRIGHT":
                continue
            if m == "ACTION:TURNLEFT":
                continue
            if m == "ACTION:FORWARD":
                continue
            if m == "EVALUATE_LOCATION":
                break
            text += m + " "
        text = text.strip()
        words = word_tokenize(text.lower())
        t_length.append(len(words))
        for word in words:
            vocab.add(word)
        nouns = 0
        adjs = 0
        preps = 0
        for t in nltk.pos_tag(words):
            pos = t[1]
            if pos in ["NN", "NNS", "NNP", "NNPS"]:
                nouns += 1
            elif pos in ["JJ", "JJR", "JJS"]:
                adjs += 1
            elif pos in ["IN", "TO"]:
                preps += 1
        avg_nouns.append(nouns)
        avg_adjs.append(adjs)
        avg_preps.append(preps)
    print("vocab size:", len(vocab))
    print("# nouns:", np.mean(avg_nouns))
    print("# adj:", np.mean(avg_adjs))
    print("# prep:", np.mean(avg_preps))


# preposition
if __name__ == "__main__":
    vocab = set()
    avg_nouns = []
    avg_adjs = []
    avg_preps = []
    t_length = []

    """WAY"""
    print("WAY analysis")
    way_data = "path/to/way"
    do_way_data("train_data.json")
    do_way_data("valSeen_data.json")
    do_way_data("valUnseen_data.json")
    do_way_data("test_data.json")

    """R2R"""
    print("R2R analysis")
    vln_data = "path/to/tasks/R2R/data/"
    do_vln_data("R2R_train.json")
    do_vln_data("R2R_val_seen.json")
    do_vln_data("R2R_val_unseen.json")
    do_vln_data("R2R_test.json")
    print("t len:", np.mean(t_length))

    """CVDN"""
    print("CVDN analysis")
    cvdn_data = "path/to/tasks/CVDN/tasks/NDH/data/"
    do_cvdn_data("train.json")
    do_cvdn_data("val_seen.json")
    do_cvdn_data("val_unseen.json")
    do_cvdn_data("test.json")
    print("t len:", np.mean(t_length))

    """TtW"""
    print("TtW analysis")
    ttw_data = "path/to/tasks/TtW/"
    do_ttw_data("talkthewalk.train.json")
    do_ttw_data("talkthewalk.test.json")
    print("t len:", np.mean(t_length))
