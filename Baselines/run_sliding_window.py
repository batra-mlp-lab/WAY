from sliding_loader import Loader
import json
import numpy as np
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from skimage.metrics import structural_similarity

"""
DESCRIPTION:
This file performs a sliding window heuristic baseline approach to the LED task.
run 'sliding_window.py' 
"""


def load_data():
    mesh2meters_file = "../data/floorplans/pix2meshDistance.json"
    data_dir = "../data/way_splits/"
    image_dir = "../data/floorplans/"
    loader = Loader(
        mesh2meters_file=mesh2meters_file,
        data_dir=data_dir,
        image_dir=image_dir,
    )
    loader.build_dataset(file="train_data.json")
    loader.build_dataset(file="valSeen_data.json")
    loader.build_dataset(file="valUnseen_data.json")
    return loader


def find_best_dialog_match(reference):
    best_score = 0
    best_text = 0
    best_index = 0
    for enum, candidate in enumerate(loader.datasets["train"].texts):
        score = sentence_bleu(reference, candidate)
        if score > best_score:
            best_score = score
            best_text = candidate
            best_index = enum
    return best_score, best_text, best_index


def loop_test(data, outfile):
    matches = []
    for reference in data.texts:
        best_score, best_text, best_index = find_best_dialog_match([reference])
        matches.append([best_score, best_text, best_index])
    np.save(outfile, matches)


def find_best_image_match(best_index, img_test, m_c_test):
    image_test = np.asarray(Image.open(img_test))[:, :, :3]
    stepSize = int(m_c_test / 2)
    windowSize = int(m_c_test)

    location = (
        loader.datasets["train"].locations[best_index][1],
        loader.datasets["train"].locations[best_index][0],
    )  # flip coordinates
    mesh_conversion = loader.datasets["train"].mesh_conversions[best_index] * 3.0
    img = Image.open(loader.datasets["train"].image_paths[best_index])
    # crop image
    img_crop_area = (
        location[1] - mesh_conversion,
        location[0] - mesh_conversion,
        location[1] + mesh_conversion,
        location[0] + mesh_conversion,
    )
    img_crop_train = img.crop(img_crop_area)
    img_crop_train = np.asarray(img_crop_train.resize((windowSize, windowSize)))
    img_crop_train = img_crop_train[:, :, :3]

    best_score = 0
    best_loc = [0, 0]
    for y in range(0, image_test.shape[0], stepSize):
        for x in range(0, image_test.shape[1], stepSize):
            patch = image_test[y : y + windowSize, x : x + windowSize]
            if patch.shape != img_crop_train.shape:
                continue
            s = structural_similarity(img_crop_train, patch, multichannel=True)
            if s > best_score:
                best_score = s
                best_loc = [y + stepSize, x + stepSize]
    return best_loc


def loop_data(data, infile, outfile):
    matches = np.load(infile, allow_pickle=True)
    distances = []

    count = 0

    for match, locations, mesh_conversion, img_test in zip(
        matches, data.locations, data.mesh_conversions, data.image_paths
    ):
        count += 1
        if count % 10 == 0:
            print(count / len(matches))
        best_score, best_text, best_index = match
        target_coord = locations[1], locations[0]
        pred_coord = find_best_image_match(best_index, img_test, mesh_conversion)
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (mesh_conversion)
        distances.append(dist)
    print("AVG DIST", np.mean(distances))
    print("ACCURACY", np.mean(np.where(np.asarray(distances) <= 3, 1, 0)))
    print("ACCURACY", np.mean(np.where(np.asarray(distances) <= 3, 1, 0)))
    x = np.where(np.asarray(distances) <= 3, 1, 0)
    print("std 3", np.std(x) / np.sqrt(len(x)))
    x = np.where(np.asarray(distances) <= 5, 1, 0)
    print("std 5", np.std(x) / np.sqrt(len(x)))
    np.save(outfile, distances)


if __name__ == "__main__":
    loader = load_data()

    """find the best train dialog match for the test dialog"""
    loop_test(loader.datasets["valSeen"], "val_seen_matches.npy")
    loop_test(loader.datasets["valUnseen"], "val_unseen_matches.npy")

    """find the best test image patch match for the test dialog"""
    loop_data(
        loader.datasets["valSeen"], "val_seen_matches.npy", "val_seen_distances.npy"
    )
    loop_data(
        loader.datasets["valUnseen"],
        "val_unseen_matches.npy",
        "val_unseen_distances.npy",
    )
