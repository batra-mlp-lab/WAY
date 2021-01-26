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
)
from graph import get_geo_dist


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

    waypoints = json.load(open(args.image_dir + "allScans_Node2pix.json"))

    distances = {"valSeen": [], "valUnseen": []}
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
                preds_way = []
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
                            wp
                            for wp, v in waypoints[scan_names[i]].items()
                            if int(v[2]) == int(levels[i])
                            and wp in scan_graphs[name][scan_names[i]].nodes()
                        ]
                        choice = candidates[np.random.randint(len(candidates))]
                        preds_way.append(choice)

                batch_locations = add_epsilon(batch_locations)
                if mode == "waypoints":  # geodesic
                    distances[name].extend(
                        geo_dist_singlefloor(
                            scan_graphs[name],
                            preds_way,
                            info_elem,
                        )
                    )
                else:  # euclidean
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

        for threshold in [0.0, 3.0, 5.0, 10.0]:
            print(
                f"\t{threshold}m-Acc: {round(accuracy(torch.tensor(dists), threshold=threshold), 3)}"
            )
            lst = accuracy_all(torch.tensor(dists), threshold=threshold)
            print(f"\t{threshold}m-SE: {round(np.std(lst) / np.sqrt(len(lst)), 3)}")
        print()


def geo_dist_singlefloor(scan_graphs, preds, info_elem):
    """Calculate distances between model predictions and targets within a batch."""
    _, _, scan_names, _, true_viewpoints = info_elem
    distances = []
    for pred_viewpoint, sn, tv in zip(preds, scan_names, true_viewpoints):
        G = scan_graphs[sn]
        distances.append(get_geo_dist(G, pred_viewpoint, tv))
    return distances


if __name__ == "__main__":
    args = cfg.parse_args()
    loader = load()
    print("testing waypoint - geodesic distance")
    baseline(args, loader, mode="waypoints")
    print("testing random - euclidean")
    baseline(args, loader, mode="random")
    print("testing center - euclidean")
    baseline(args, loader, mode="center")
