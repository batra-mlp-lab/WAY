import torch
import torch.nn as nn
import numpy as np
import json
from utils import accuracy, accuracy_batch
from graph import snap_to_grid


def format_preds_singlefloor_output(
    node2pix, scan_graphs, preds, mesh_conversions, info_elem, predictions_dir, mode
):
    saveData = {}
    _, levels, scan_names, ann_ids, _ = info_elem
    preds = preds.cpu()
    preds = nn.functional.interpolate(
        preds.unsqueeze(1), (700, 1200), mode="bilinear"
    ).squeeze(1)
    for pred, conversion, level, sn, ann_id in zip(
        preds, mesh_conversions, levels, scan_names, ann_ids
    ):
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        G = scan_graphs[sn]
        pred_viewpoint = snap_to_grid(G, node2pix, sn, pred_coord, conversion, level)
        saveData[ann_id] = {
            "viewpoint": pred_viewpoint,
        }
    with open(predictions_dir + mode + "_predictions.json", "w") as f:
        json.dump(saveData, f)


def format_preds_multifloor_output(
    node2pix, scan_graphs, preds, mesh_conversions, info_elem, predictions_dir, mode
):
    saveData = {}
    _, _, scan_names, ann_ids, _ = info_elem
    for pred, conversion, sn, ann_id in zip(
        preds, mesh_conversions, scan_names, ann_ids
    ):
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()]))
        pred = pred.cpu()
        pred = nn.functional.interpolate(
            pred.unsqueeze(1), (700, 1200), mode="bilinear"
        ).squeeze(1)[:total_floors]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        G = scan_graphs[sn]
        convers = conversion.view(5, 1, 1)[pred_coord[0].item()]
        pred_viewpoint = snap_to_grid(
            G,
            node2pix,
            sn,
            [pred_coord[1].item(), pred_coord[2].item()],
            convers,
            pred_coord[0].item(),
        )
        saveData[ann_id] = {
            "viewpoint": pred_viewpoint,
        }
    with open(predictions_dir + mode + "_predictions.json", "w") as f:
        json.dump(saveData, f)


def eval_preds(errors):
    for split, dists in errors.items():
        print(f"{split}")
        print(f"\tLE: {round(np.mean(dists), 2)}")
        print(f"\tSE: {round(np.std(dists) / np.sqrt(len(dists)), 2)}")
        print(f"\tSD: {round(np.std(dists), 2)}")

        for threshold in [0.0, 3.0, 5.0, 10.0]:
            print(
                f"\t{threshold}m-Acc: {round(accuracy(torch.tensor(dists), threshold=threshold), 3)}"
            )
            lst = accuracy_batch(torch.tensor(dists), threshold=threshold)
            print(f"\t{threshold}m-SE: {round(np.std(lst) / np.sqrt(len(lst)), 3)}")
        print()