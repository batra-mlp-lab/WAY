import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
import random
import numpy as np
import json
import sys

sys.path.insert(1, "../lingUnet")
import cfg
from loader import Loader
from utils import (
    accuracy,
    add_epsilon,
    euclidean_dist_singlefloor,
    geo_dist_singlefloor,
)


def load():
    loader = Loader(
        mesh2meters_file=args.mesh2meters,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
    )
    loader.build_dataset(file="valSeen_data.json", args=args)
    loader.build_dataset(file="valUnseen_data.json", args=args)
    return loader


def accuracy_all(dists, threshold):
    return (dists <= threshold).int().numpy().tolist()


def baseline(args, loader, mode="random"):
    r"""Mode can either be random (by pixel) or center, which always predicts the center pixel.
    Repeats the random experiment 5 times and take the average. the center baseline is deterministic so it runs once.
    """

    valSeen_iterator = DataLoader(
        dataset=loader.datasets["valSeen"], batch_size=50, shuffle=False
    )
    valUnseen_iterator = DataLoader(
        dataset=loader.datasets["valUnseen"], batch_size=50, shuffle=False
    )
    scan_graphs = {}
    scan_graphs["valSeen"] = loader.datasets["valSeen"].scan_graphs
    scan_graphs["valUnseen"] = loader.datasets["valUnseen"].scan_graphs

    allScans_Node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))
    waypoints = {}
    for k, v in allScans_Node2pix.items():
        waypoints[k] = [v[i] for i in v]
    px_downSampleH = 700
    px_downSampleW = 1200

    distances = {"valUnseen": [], "valSeen": []}
    for i in range(5 if mode == "random" else 1):
        for name, split_iterator in zip(
            ["valSeen", "valUnseen"],
            [valSeen_iterator, valUnseen_iterator],
        ):
            for (
                _,
                info_elem,
                batch_images,
                batch_texts,
                batch_seq_lengths,
                batch_locations,
                batch_conversions,
                _,
                _,
            ) in tqdm.tqdm(split_iterator):
                dialogs, levels, scan_names, ann_ids, _ = info_elem
                bs, H, W = batch_locations.size()
                batch_locations = nn.functional.interpolate(
                    batch_locations.unsqueeze(1),
                    (args.ds_height, args.ds_width),
                    mode="bilinear",
                    align_corners=True,
                )
                batch_locations = batch_locations.squeeze(1)

                preds = torch.zeros(*batch_locations.size())
                for i in range(bs):
                    if mode == "random":
                        preds[
                            i,
                            random.randint(0, preds.size(1)) - 1,
                            random.randint(0, preds.size(2)) - 1,
                        ] = 1.0
                    elif mode == "center":
                        preds[i, int(preds.size(1) / 2), int(preds.size(2) / 2)] = 1.0
                    elif mode == "waypoints":
                        candidates = [
                            wp[0]
                            for wp in waypoints[scan_names[i]]
                            if int(wp[2]) == int(levels[i])
                        ]
                        choice = candidates[np.random.randint(len(candidates))]
                        choice[0] = int(args.ds_width * choice[0] / px_downSampleW)
                        choice[1] = int(args.ds_height * choice[1] / px_downSampleH)

                        preds[i, choice[1], choice[0]] = 1.0

                batch_locations = add_epsilon(batch_locations)
                if args.distance_metric == "geodesic":
                    distances[name].extend(
                        geo_dist_singlefloor(
                            allScans_Node2pix,
                            scan_graphs[name],
                            preds,
                            batch_conversions,
                            info_elem,
                        )
                    )
                elif args.distance_metric == "euclidean":
                    distances[name].extend(
                        euclidean_dist_singlefloor(
                            preds, batch_locations, batch_conversions, H, W
                        )
                    )

    for split, dists in distances.items():
        print(f"{split}: {mode}")
        print(f"\tLE: {round(np.mean(dists), 2)}")
        print(f"\tSE: {round(np.std(dists) / np.sqrt(len(dists)), 2)}")
        print(f"\tSD: {round(np.std(dists), 2)}")

        for threshold in [1.0, 3.0, 5.0, 10.0]:
            print(
                f"\t{threshold}m-Acc: {round(accuracy(torch.tensor(dists), threshold=threshold), 3)}"
            )
            lst = accuracy_all(torch.tensor(dists), threshold=threshold)
            print(f"\t{threshold}m-SE: {round(np.std(lst) / np.sqrt(len(lst)), 3)}")
        print()


if __name__ == "__main__":
    args = cfg.parse_args()
    loader = load()
    print("testing random")
    baseline(args, loader, mode="random")
    print("testing center")
    baseline(args, loader, mode="center")
    print("testing waypoint")
    baseline(args, loader, mode="waypoints")