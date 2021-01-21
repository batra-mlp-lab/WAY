import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from collections import defaultdict
from nltk.tokenize import word_tokenize
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
import ipdb
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from nltk.tokenize import word_tokenize


class Loader:
    def __init__(self, mesh2meters_file, data_dir, image_dir):
        self.mesh2meters = json.load(open(mesh2meters_file))
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.datasets = {}

    def load_image_paths(self, data):
        image_paths = []
        scan_names = []
        levels = []
        for data_obj in data:
            scan_name = data_obj["scanName"]
            scan_names.append(scan_name)
            level = str(data_obj["finalLocation"]["floor"])
            levels.append(level)
            image_paths.append(
                "{}floor_{}/{}_{}.png".format(self.image_dir, level, scan_name, level)
            )
        return image_paths, scan_names, levels

    def add_tokens(self, message_arr):
        new_dialog = ""
        for enum, message in enumerate(message_arr):
            if enum % 2 == 0:
                new_dialog += " " + message + " "
            else:
                new_dialog += message
        new_dialog = new_dialog.strip()
        new_dialog = word_tokenize(new_dialog.lower())
        return new_dialog

    def load_dialogs(self, data):
        dialogs = []
        for data_obj in data:
            new_dialog = self.add_tokens(data_obj["dialogArray"])
            dialogs.append(new_dialog)
        return dialogs

    def load_locations(self, data):
        return [data_obj["finalLocation"]["pixel_coord"] for data_obj in data]

    def load_mesh_conversion(self, data):
        mesh_conversions = []
        for data_obj in data:
            floor = str(data_obj["finalLocation"]["floor"])
            oneMRadius = (
                self.mesh2meters[data_obj["scanName"]][floor]["threeMeterRadius"] / 3.0
            )
            mesh_conversions.append(oneMRadius)
        return mesh_conversions

    def build_dataset(
        self,
        file,
    ):
        mode = file.split("_")[0]
        print("[{}]: Start loading JSON file...".format(mode))
        data = json.load(open(self.data_dir + file))
        locations = self.load_locations(data)
        mesh_conversions = self.load_mesh_conversion(data)
        image_paths, scan_names, levels = self.load_image_paths(data)

        print("[{}]: Building vocab from text data...".format(mode))
        texts = self.load_dialogs(data)

        print("[{}]: Building dataset...".format(mode))
        dataset = LEDDataset(
            image_paths,
            texts,
            mesh_conversions,
            locations,
            scan_names,
            levels,
        )
        self.datasets[mode] = dataset
        print("[{}]: Finish building dataset...".format(mode))


class LEDDataset(Dataset):
    def __init__(
        self,
        image_paths,
        texts,
        mesh_conversions,
        locations,
        scan_names,
        levels,
    ):
        self.image_paths = image_paths
        self.texts = texts
        self.locations = locations
        self.mesh_conversions = mesh_conversions
        self.scan_names = scan_names
        self.levels = levels
