import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from nltk.tokenize import word_tokenize
import numpy as np
import copy
import random

from led_dataset import LEDDataset


class Loader:
    def __init__(self, mesh2meters_file, data_dir, image_dir):
        self.mesh2meters = json.load(open(mesh2meters_file))
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.vocab = Vocabulary()
        self.max_length = 0
        self.max_dialog_length = 0
        self.datasets = {}
        self.args = None

    def load_image_paths(self, data):
        image_paths = []
        scan_names = []
        levels = []
        annotation_ids = []
        for data_obj in data:
            scan_name = data_obj["scanName"]
            scan_names.append(scan_name)
            annotation_ids.append(data_obj["annotationId"])
            level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)

            image_paths.append(
                "{}floor_{}/{}_{}.png".format(self.image_dir, level, scan_name, level)
            )
        return image_paths, scan_names, levels, annotation_ids

    def create_dialog_pairs(self, message_arr):
        dialog_pairs = []
        for i in range(len(message_arr)):
            if i % 2 == 1:
                dialog_pairs.append(
                    "SOLM "
                    + message_arr[i - 1]
                    + " EOLM SOOM "
                    + message_arr[i]
                    + " EOOM"
                )
        if len(message_arr) % 2 == 1:
            dialog_pairs.append("SOLM " + message_arr[-1] + " EOLM SOOM EOOM")
        return dialog_pairs

    def add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 0:
                new_dialog += "SOLM " + message + " EOLM "
            else:
                new_dialog += "SOOM " + message + " EOOM "
        return new_dialog

    def load_dialogs(self, data):
        dialogs = []
        for data_obj in data:
            if self.args.language_change == "shuffle":
                dialog_pairs = self.create_dialog_pairs(data_obj["dialogArray"])
                random.shuffle(dialog_pairs)
                new_dialog = " ".join(dialog_pairs)

            elif self.args.language_change == "locator_only":
                dialogArray = []
                for enum, i in enumerate(data_obj["dialogArray"]):
                    if enum % 2 == 0:
                        dialogArray.append(i)
                new_dialog = "SOLM " + " EOLM SOLM ".join(dialogArray) + " EOLM"

            elif self.args.language_change == "observer_only":
                dialogArray = []
                for enum, i in enumerate(data_obj["dialogArray"]):
                    if enum % 2 == 1:
                        dialogArray.append(i)
                new_dialog = "SOOM " + " EOOM SOOM ".join(dialogArray) + " EOOM"

            elif self.args.language_change == "first_half":
                mid_point = int(round(len(data_obj["dialogArray"]) / 2))
                if mid_point % 2 == 1:
                    mid_point += 1
                new_dialog = self.add_tokens(data_obj["dialogArray"][:mid_point])

            elif self.args.language_change == "second_half":
                mid_point = int(round(len(data_obj["dialogArray"]) / 2))
                if mid_point % 2 == 1:
                    mid_point -= 1
                new_dialog = self.add_tokens(data_obj["dialogArray"][mid_point:])

            else:  # language_change == 'none'
                new_dialog = self.add_tokens(data_obj["dialogArray"])

            dialogs.append(new_dialog)
        return dialogs

    def load_locations(self, data, mode):
        if "test" in mode:
            return [[0, 0] for data_obj in data], ["" for data_obj in data]

        x = [
            [
                data_obj["finalLocation"]["pixel_coord"][1],
                data_obj["finalLocation"]["pixel_coord"][0],
            ]
            for data_obj in data
        ]

        y = [data_obj["finalLocation"]["viewPoint"] for data_obj in data]

        return x, y

    def load_mesh_conversion(self, data):
        mesh_conversions = []
        for data_obj in data:
            mesh_conversions.append(
                self.mesh2meters[data_obj["scanName"]][
                    str(data_obj["finalLocation"]["floor"])
                ]["threeMeterRadius"]
                / 3.0
            )
        return mesh_conversions

    def build_vocab(self, texts, mode):
        """Add words to the vocabulary"""
        ids = []
        seq_lengths = []
        for text in texts:
            line_ids = []
            words = word_tokenize(text.lower())
            self.max_length = max(self.max_length, len(words))
            for word in words:
                word = self.vocab.add_word(word, mode)
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_pretrained_vocab(self, texts):
        self.vocab.word2idx = json.load(open(self.args.embedding_dir + "word2idx.json"))
        self.vocab.idx2word = json.load(open(self.args.embedding_dir + "idx2word.json"))
        ids = []
        seq_lengths = []
        for text in texts:
            line_ids = []
            words = word_tokenize(text.lower())
            self.max_length = max(self.max_length, len(words))
            for word in words:
                line_ids.append(self.vocab.word2idx[word])
            ids.append(line_ids)
            seq_lengths.append(len(words))
        text_ids = np.array([row + [0] * (self.max_length - len(row)) for row in ids])
        return text_ids, seq_lengths

    def build_dataset(self, file, args):
        self.args = args
        mode = file.split("_")[0]
        print("[{}]: Loading JSON file...".format(mode))
        data = json.load(open(self.data_dir + file))

        if isinstance(args.sample_used, tuple):
            start, end = args.sample_used
            data = data[start:end]
            num_samples = end - start
        elif isinstance(args.sample_used, float):
            num_samples = int(len(data) * args.sample_used)
            data = data[:num_samples]
        print(
            "[{}]: Using {} ({}%) samples".format(
                mode, num_samples, num_samples / len(data) * 100
            )
        )

        locations, viewPoint_location = self.load_locations(data, mode)
        mesh_conversions = self.load_mesh_conversion(data)
        image_paths, scan_names, levels, annotation_ids = self.load_image_paths(data)

        print(
            "[{}]: Building Vocab, Language Change is {}".format(
                mode, args.language_change
            )
        )
        dialogs = self.load_dialogs(data)
        texts = copy.deepcopy(dialogs)
        if (
            self.args.embedding_type == "glove"
            or self.args.embedding_type == "word2vec"
        ):
            texts, seq_lengths = self.build_pretrained_vocab(texts)
        else:
            texts, seq_lengths = self.build_vocab(texts, mode)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            args,
            image_paths,
            texts,
            seq_lengths,
            mesh_conversions,
            locations,
            viewPoint_location,
            dialogs,
            scan_names,
            levels,
            annotation_ids,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))

    def build_dataset_extra_train(self, file, args):
        self.args = args
        mode = file.split("_")[0]
        print("[{}]: Start loading JSON file...".format(mode))
        data = json.load(open(self.data_dir + file))
        if isinstance(args.sample_used, tuple):
            start, end = args.sample_used
            data = data[start:end]
            num_samples = end - start
        elif isinstance(args.sample_used, float):
            num_samples = int(len(data) * args.sample_used)
            data = data[:num_samples]
        print(
            "[{}]: Using {} ({}%) samples".format(
                mode, num_samples, num_samples / len(data) * 100
            )
        )

        # load data
        locations, viewPoint_location, mesh_conversions, image_paths = [], [], [], []
        scan_names, levels, dialogs, annotation_ids = [], [], [], []
        for data_obj in data:
            sn = data_obj["scanName"]
            ann_id = data_obj["annotationId"]
            for enum, p in enumerate(data_obj["detailedNavPath"]):
                pixel_coord = [p[-1][1][1], p[-1][1][0]]
                floor = str(p[-1][2])
                locations.append(pixel_coord)
                viewPoint_location.append(p[-1][0])
                oneMRadius = self.mesh2meters[sn][floor]["threeMeterRadius"] / 3.0
                mesh_conversions.append(oneMRadius)
                scan_names.append(sn)
                annotation_ids.append(ann_id)
                levels.append(floor)
                image_paths.append(
                    "{}floor_{}/{}_{}.png".format(self.image_dir, floor, sn, floor)
                )
                dialogs.append(
                    self.add_tokens(data_obj["dialogArray"][: (enum + 1) * 2])
                )
            if pixel_coord != data_obj["finalLocation"]["pixel_coord"]:
                floor = str(data_obj["finalLocation"]["floor"])
                locations.append(
                    [
                        data_obj["finalLocation"]["pixel_coord"][1],
                        data_obj["finalLocation"]["pixel_coord"][0],
                    ]
                )
                viewPoint_location.append(data_obj["finalLocation"]["viewPoint"])
                oneMRadius = self.mesh2meters[sn][floor]["threeMeterRadius"] / 3.0
                mesh_conversions.append(oneMRadius)
                scan_names.append(sn)
                annotation_ids.append(ann_id)
                levels.append(floor)
                image_paths.append(
                    "{}floor_{}/{}_{}.png".format(self.image_dir, floor, sn, floor)
                )
                dialogs.append(self.add_tokens(data_obj["dialogArray"]))

        print("[{}]: Building vocab from text data...".format(mode))
        texts = copy.deepcopy(dialogs)
        if args.embedding_type == "glove" or args.embedding_type == "word2vec":
            texts, seq_lengths = self.build_pretrained_vocab(texts, args.embedding_dir)
        else:
            texts, seq_lengths = self.build_vocab(texts, mode)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            mode,
            args,
            image_paths,
            texts,
            seq_lengths,
            mesh_conversions,
            locations,
            viewPoint_location,
            dialogs,
            scan_names,
            levels,
            annotation_ids,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}

    def add_word(self, word, mode):
        if word not in self.word2idx and mode in ("train"):
            idx = len(self.idx2word)
            self.idx2word[idx] = word
            self.word2idx[word] = idx
            return word
        elif word not in self.word2idx and mode != "train":
            return "<unk>"
        else:
            return word

    def __len__(self):
        return len(self.idx2word)
