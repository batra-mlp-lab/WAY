import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import os
import torch.nn as nn
import math

from graph import get_geo_dist, snap_to_grid


def geo_dist_singlefloor(node2pix, scan_graphs, preds, mesh_conversions, info_elem):
    """Calculate distances between model predictions and targets within a batch."""
    H, W = 700, 1200
    _, levels, scan_names, _, true_viewpoints = info_elem
    preds = preds.cpu()
    preds = nn.functional.interpolate(
        preds.unsqueeze(1),
        (H, W),
        mode="bilinear",
        align_corners=True,
    ).squeeze(1)

    distances = []
    for pred, conversion, level, sn, tv in zip(
        preds, mesh_conversions, levels, scan_names, true_viewpoints
    ):
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        G = scan_graphs[sn]
        pred_viewpoint = snap_to_grid(
            G, node2pix, sn, pred_coord, conversion, level, tv
        )
        distances.append(get_geo_dist(G, pred_viewpoint, tv))
    return distances


def geo_dist_multifloor(node2pix, scan_graphs, preds, mesh_conversions, info_elem):
    """Calculate distances between model predictions and targets within a batch."""
    distances = []
    _, _, scan_names, _, true_viewpoints = info_elem
    for pred, conversion, sn, tv in zip(
        preds,
        mesh_conversions,
        scan_names,
        true_viewpoints,
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
            tv,
        )
        distances.append(get_geo_dist(G, pred_viewpoint, tv))
    return distances


def euclidean_dist_singlefloor(preds, targets, mesh_conversions, H, W):
    """Calculate distances between model predictions and targets within a batch."""
    preds = preds.cpu()
    targets = targets.cpu()
    distances = []
    targets = nn.functional.interpolate(
        targets.unsqueeze(1),
        (H, W),
        mode="bilinear",
        align_corners=True,
    ).squeeze(1)
    preds = nn.functional.interpolate(
        preds.unsqueeze(1),
        (H, W),
        mode="bilinear",
        align_corners=True,
    ).squeeze(1)

    for pred, target, conversion in zip(preds, targets, mesh_conversions):
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        target_coord = np.unravel_index(target.argmax(), target.size())
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (conversion)
        distances.append(dist.item())
    return distances


def euclidean_dist_multifloor(node2pix, preds, targets, conversions, info_elem, H, W):
    """Calculate distances between model predictions and targets within a batch."""
    distances = []
    _, true_levels, scan_names, _, _ = info_elem
    for pred, convers, target, true_level, sn in zip(
        preds, conversions, targets, true_levels, scan_names
    ):  # for batches
        true_level = int(true_level)
        total_floors = len(set([v[2] for k, v in node2pix[sn].items()]))
        pred = pred.cpu()
        pred = nn.functional.interpolate(
            pred.unsqueeze(1),
            (H, W),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)[:total_floors]
        convers = convers.view(5, 1, 1)[true_level]
        pred_coord = np.unravel_index(pred.argmax(), pred.size())
        if true_level != pred_coord[0]:
            distances.append(11)
        target = target.cpu()
        target_coord = np.unravel_index(target.argmax(), target.size())

        dist = np.sqrt(
            (target_coord[0] - pred_coord[1]) ** 2
            + (target_coord[1] - pred_coord[2]) ** 2
        ) / (convers)
        distances.append(dist.item())

    return distances


def add_epsilon(locations):
    og_size = locations.size()
    locations = locations + 1.0e-5
    locations = locations.view(og_size[0], -1)
    locations = locations / locations.sum(-1, keepdim=True)
    locations = locations.view(og_size)
    return locations


def accuracy(dists, threshold):
    """Calculating accuracy at 3 meters by default"""
    return np.mean((torch.tensor(dists) <= threshold).int().numpy())


def accuracy_batch(dists, threshold):
    return (dists <= threshold).int().numpy().tolist()


def convert_model_to_state(model, args, rnn_args, cnn_args):
    state = {
        "args": vars(args),
        "rnn_args": rnn_args,
        "cnn_args": cnn_args,
        "out_layer_args": {},
        "state_dict": {},
    }
    # use copies instead of references
    for k, v in model.state_dict().items():
        state["state_dict"][k] = v.clone().to(torch.device("cpu"))
    return state


def load_oldArgs(args, oldArgs):
    args.m = oldArgs["num_lingunet_layers"]
    args.image_channels = oldArgs["linear_hidden_size"]
    args.blind_lang = oldArgs["blind_lang"]
    args.blind_vis = oldArgs["blind_vis"]
    args.freeze_resnet = oldArgs["freeze_resnet"]
    args.res_connect = oldArgs["res_connect"]
    args.avgpool = oldArgs["avgpool"]

    args.embed_size = oldArgs["embed_size"]
    args.rnn_hidden_size = oldArgs["rnn_hidden_size"]
    args.num_rnn_layers = oldArgs["num_rnn_layers"]
    args.embed_dropout = oldArgs["embed_dropout"]
    args.bidirectional = oldArgs["bidirectional"]
    args.embedding_type = oldArgs["embedding_type"]

    args.linear_hidden_size = oldArgs["linear_hidden_size"]
    args.num_linear_hidden_layers = oldArgs["num_linear_hidden_layers"]

    args.ds_percent = oldArgs["ds_percent"]
    args.language_change = oldArgs["language_change"]
    args.ds_height = oldArgs["ds_height"]
    args.ds_width = oldArgs["ds_width"]

    return args


def log(args, mode, log_info):
    if args.train == True:
        if mode == "train":
            log_string = "[Train]: Epoch {:3d} | lr {:04.4f} | Loss {:5.6f} | Mean Dist {:5.4f} | Accuracy {:5.4f}"
        elif mode == "valSeen":
            log_string = "[ValSeen]: Epoch {:3d} | Loss {:5.6f} | Mean Dist {:5.4f} | Accuracy {:5.4f}"
        elif mode == "valUnseen":
            log_string = "[ValUnseen]: Epoch {:3d} | Loss {:5.6f} | Mean Dist {:5.4f} | Accuracy {:5.4f}"
    else:
        if mode == "valSeen":
            log_string = "[ValSeen]: Mean Dist {:5.4f} | Accuracy {:5.4f}"
        elif mode == "valUnseen":
            log_string = "[ValUnseen]: Mean Dist {:5.4f} | Accuracy {:5.4f}"
        elif mode == "test":
            log_string = "[Test]: Mean Dist {:5.4f} | Accuracy {:5.4f}"
    print(log_string.format(*log_info))


def plot_err_v_env_size(args, floors, scans, distances, viz_dir, mode):
    distances = [x.item() for x in distances]

    scanVdist = {}
    for dist, scan in zip(distances, scans):
        if scan in scanVdist:
            scanVdist[scan].append(dist)
        else:
            scanVdist[scan] = [dist]
    for key, value in scanVdist.items():
        scanVdist[key] = np.mean(np.asarray(value))
    plt.bar(range(len(scanVdist)), list(scanVdist.values()), align="center")
    plt.title(mode)
    plt.xticks(range(len(scanVdist)), list(scanVdist.keys()))
    plt.xticks(rotation=90, fontsize=10)
    plt.ylabel("Localization Error (meters)")
    plt.show()
    plt.savefig(viz_dir + mode + "_errVenv.png", bbox_inches="tight")

    mesh2meters = json.load(open(args.node2pix_file))
    envSize = []
    for scan, floor in zip(scans, floors):
        envSize.append(mesh2meters[scan][floor]["threeMeterRadius"])
    plt.plot(envSize, distances, "ro")
    plt.title(mode)
    plt.xlabel("#Pixels : 3 meters")
    plt.ylabel("Localization Error (meters)")
    plt.show()
    plt.savefig(viz_dir + mode + "_errVpixelmeterratio.png", bbox_inches="tight")


def visualize(args, info_elem, housemaps, targets, preds, errors):
    dialogs, levels, scan_names, ann_ids, _ = info_elem
    saveData = {
        "annotation_ids": ann_ids,
        "dialogs": dialogs,
        "errors": errors,
    }
    json.dump(saveData, open(args.visualization_dir + "sample_predictions.json", "w"))

    for housemap, target, pred, ann_id in zip(housemaps, targets, preds, ann_ids):
        housemap = housemap.numpy()
        pred = nn.functional.interpolate(
            pred.unsqueeze(0).unsqueeze(0), (housemap.shape[0], housemap.shape[1])
        ).squeeze()
        pred = pred.cpu().detach().numpy()
        target = nn.functional.interpolate(
            target.unsqueeze(0).unsqueeze(0), (housemap.shape[0], housemap.shape[1])
        ).squeeze()
        target = target.cpu().detach().numpy()
        f = plt.figure()
        f.clear()
        plt.axis("off")
        plt.imshow(target, alpha=0.5)
        plt.imshow(housemap, alpha=0.5)
        plt.savefig(
            args.visualization_dir + ann_id + "_target.png",
            bbox_inches="tight",
        )
        f.clear()
        plt.axis("off")
        plt.imshow(housemap, alpha=0.5)
        plt.imshow(pred, alpha=0.5)
        plt.savefig(
            args.visualization_dir + ann_id + "_prediction.png",
            bbox_inches="tight",
        )


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(a, dim, order_index)