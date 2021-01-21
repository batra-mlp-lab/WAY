import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torchvision.models as models

import tqdm
import numpy as np
import os.path
import logging
import argparse
import datetime
import os
import copy
import cfg
from PIL import Image
import json

from loader import Loader
from lingunet_model import LingUNet
from utils import *
from evaluate import *


class LingUNetAgent:
    def __init__(self, args):
        self.args = args
        self.device = (
            torch.device(f"cuda:{args.cuda}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.args.device = self.device
        self.loss_func = nn.KLDivLoss(reduction="batchmean")

        self.loader = None
        self.scan_graphs = {}
        self.writer = None
        self.out_dir = os.path.join(args.log_dir, args.run_name)
        if (args.log and args.train) or args.save:
            if not os.path.isdir(self.out_dir):
                print("Log directory under {}".format(self.out_dir))
                os.system("mkdir {}".format(self.out_dir))
            self.writer = SummaryWriter(args.summary_dir + args.name)

        self.model = None
        self.optimizer = None
        self.node2pix = json.load(open(args.image_dir + "allScans_Node2pix.json"))

    def testEpochMultiFloor(self, data_iterator, mode, epoch=0):
        self.model.eval()
        distances = []
        with torch.no_grad():
            for batch_data in tqdm.tqdm(data_iterator):
                (
                    _,
                    info_elem,
                    batch_images,
                    batch_texts,
                    batch_seq_lengths,
                    batch_locations,
                    _,
                    all_maps,
                    all_conversions,
                ) = batch_data

                # get predictions, calculate loss
                batch_texts = tile(batch_texts, dim=0, n_tile=self.args.max_floors)
                batch_seq_lengths = tile(
                    batch_seq_lengths, dim=0, n_tile=self.args.max_floors
                )

                preds = self.model(
                    all_maps.view(
                        (
                            -1,
                            all_maps.size()[2],
                            all_maps.size()[3],
                            all_maps.size()[4],
                        )
                    ).to(device=self.args.device),
                    batch_texts.to(device=self.args.device),
                    batch_seq_lengths.to(device=self.args.device),
                ).view(
                    all_maps.size(0),
                    self.args.max_floors,
                    self.args.ds_height,
                    self.args.ds_width,
                )

                # calculate location error and accuracy
                if "test" not in mode:
                    if self.args.distance_metric == "geodesic":
                        distances.extend(
                            geo_dist_multifloor(
                                self.node2pix,
                                self.scan_graphs[mode],
                                preds,
                                all_conversions,
                                info_elem,
                            )
                        )
                    elif self.args.distance_metric == "euclidean":
                        _, _, H, W = batch_images.size()
                        distances.extend(
                            euclidean_dist_multifloor(
                                self.node2pix,
                                preds,
                                batch_locations,
                                all_conversions,
                                info_elem,
                                H,
                                W,
                            )
                        )

            if "test" not in mode:
                avg_dist = np.mean(distances)
                avg_acc = accuracy(distances, 3)
                log(self.args, mode, (avg_dist, avg_acc))
            else:
                format_preds_multifloor_output(
                    self.node2pix,
                    self.scan_graphs[mode],
                    preds,
                    all_conversions,
                    info_elem,
                    self.args.predictions_dir,
                    mode,
                )
                return None, None, None

        return None, avg_acc, distances

    def testEpochSingleFloor(self, data_iterator, mode, epoch=0):
        self.model.eval()
        distances, total_loss = [], []

        with torch.no_grad():
            for enum, batch_data in enumerate(tqdm.tqdm(data_iterator)):
                (
                    housemaps,
                    info_elem,
                    batch_images,
                    batch_texts,
                    batch_seq_lengths,
                    batch_locations,
                    batch_conversions,
                    _,
                    _,
                ) = batch_data

                # downsample batch locations
                _, _, H, W = batch_images.size()
                batch_locations = (
                    nn.functional.interpolate(
                        batch_locations.unsqueeze(1),
                        (self.args.ds_height, self.args.ds_width),
                        mode="bilinear",
                        align_corners=True,
                    )
                    .squeeze(1)
                    .float()
                ).to(device=self.args.device)

                # get predictions
                preds = self.model(
                    batch_images.to(device=self.args.device),
                    batch_texts.to(device=self.args.device),
                    batch_seq_lengths.to(device=self.args.device),
                )

                if "test" not in mode:
                    if self.args.distance_metric == "geodesic":
                        distances.extend(
                            geo_dist_singlefloor(
                                self.node2pix,
                                self.scan_graphs[mode],
                                preds,
                                batch_conversions,
                                info_elem,
                            )
                        )
                    elif self.args.distance_metric == "euclidean":
                        distances.extend(
                            euclidean_dist_singlefloor(
                                preds, batch_locations, batch_conversions, H, W
                            )
                        )
                if self.args.train == True:
                    loss = self.loss_func(preds, add_epsilon(batch_locations))
                    total_loss.append(loss.item())
                    if self.args.visualize and enum == 0:
                        visualize(
                            args,
                            info_elem,
                            housemaps,
                            batch_locations,
                            preds,
                            distances,
                        )

        if self.args.train:
            avg_acc = accuracy(distances, 3)
            mean_dist = np.mean(distances)
            avg_loss = np.mean(total_loss)
            log(self.args, mode, (epoch, avg_loss, mean_dist, avg_acc))
            return avg_loss, avg_acc, distances
        else:
            if "test" not in mode:
                avg_acc = accuracy(distances, 3)
                mean_dist = np.mean(distances)
                log(self.args, mode, (mean_dist, avg_acc))
                return None, avg_acc, distances
            else:
                format_preds_singlefloor_output(
                    self.node2pix,
                    self.scan_graphs[mode],
                    preds,
                    batch_conversions,
                    info_elem,
                    self.args.predictions_dir,
                    mode,
                )
                return None, None, None

    def run_test(self):
        print("Starting Evaluation...")
        oldArgs, rnn_args, cnn_args, out_layer_args, state_dict = torch.load(
            self.args.eval_ckpt
        ).values()
        self.args = load_oldArgs(self.args, oldArgs)

        self.load_data()

        self.model = LingUNet(rnn_args, cnn_args, self.args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device=self.args.device)

        if self.args.test_multi_floor:
            _, _, valseen_errors = self.testEpochMultiFloor(
                self.valseen_iterator, mode="valSeen"
            )
            _, _, valunseen_errors = self.testEpochMultiFloor(
                self.val_unseen_iterator, mode="valUnseen"
            )
            self.testEpochMultiFloor(self.test_iterator, mode="test")
        else:
            _, _, valseen_errors = self.testEpochSingleFloor(
                self.valseen_iterator, mode="valSeen"
            )
            _, _, valunseen_errors = self.testEpochSingleFloor(
                self.val_unseen_iterator, mode="valUnseen"
            )
            self.testEpochSingleFloor(self.test_iterator, mode="test")
        eval_preds({"valSeen": valseen_errors, "valUnseen": valunseen_errors})

    def trainEpoch(self):
        self.model.train()
        total_loss, total_accuracy = [], []

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
        ) in tqdm.tqdm(self.train_iterator):
            self.optimizer.zero_grad()

            # downsample batch locations
            _, _, H, W = batch_images.size()
            if self.args.data_aug:
                crop_size = self.args.ds_height_crop, self.args.ds_width_crop
            else:
                crop_size = self.args.ds_height, self.args.ds_width
            batch_locations = (
                nn.functional.interpolate(
                    batch_locations.unsqueeze(1), (crop_size), mode="bilinear"
                )
                .squeeze(1)
                .float()
            ).to(device=self.args.device)

            # get predictions
            preds = self.model(
                batch_images.to(device=self.args.device),
                batch_texts.to(device=self.args.device),
                batch_seq_lengths.to(device=self.args.device),
            )

            # calculate loss
            loss = self.loss_func(preds, add_epsilon(batch_locations))
            total_loss.append(loss.item())
            loss.backward()

            # calculate location error and accuracy
            if self.args.distance_metric == "geodesic":
                total_accuracy.append(
                    accuracy(
                        geo_dist_singlefloor(
                            self.node2pix,
                            self.scan_graphs["train"],
                            preds,
                            batch_conversions,
                            info_elem,
                        ),
                        3,
                    )
                )
            elif self.args.distance_metric == "euclidean":
                total_accuracy.append(
                    accuracy(
                        euclidean_dist_singlefloor(
                            preds, batch_locations, batch_conversions, H, W
                        ),
                        3,
                    )
                )
            nn.utils.clip_grad_value_(
                self.model.parameters(), clip_value=self.args.grad_clip
            )
            self.optimizer.step()

        train_loss = np.mean(total_loss)
        train_acc = np.mean(np.asarray(total_accuracy))
        return train_acc, train_loss

    def run_train(self):
        assert self.args.num_lingunet_layers is not None
        rnn_args = {"input_size": len(self.loader.vocab)}

        cnn_args = {
            "kernel_size": 5,
            "padding": 2,
            "deconv_dropout": 0,
            "conv_dropout": 0,
        }

        self.model = LingUNet(rnn_args, cnn_args, args)

        num_params = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print("Number of parameters:", num_params)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device=self.args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        print("Starting Training...")
        best_tune_acc = float("-inf")
        best_unseen_acc = float("-inf")
        best_model = None
        patience = 0
        save_path = ""
        for epoch in range(self.args.num_epoch):
            train_acc, train_loss = self.trainEpoch()
            valseen_loss, valseen_acc, _ = self.testEpochSingleFloor(
                self.valseen_iterator, "valSeen", epoch
            )
            val_unseen_loss, val_unseen_acc, _ = self.testEpochSingleFloor(
                self.val_unseen_iterator, "valUnseen", epoch
            )

            if self.args.log:
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Accuracy/val_seen", valseen_acc, epoch)
                self.writer.add_scalar("Loss/val_seen", valseen_loss, epoch)
                self.writer.add_scalar("Accuracy/val_unseen", val_unseen_acc, epoch)
                self.writer.add_scalar("Loss/val_unseen", val_unseen_loss, epoch)

            if valseen_acc > best_tune_acc or val_unseen_acc > best_unseen_acc:
                best_model = copy.deepcopy(self.model)
                if self.args.save:
                    save_path = os.path.join(
                        self.out_dir,
                        "{}_acc{:.2f}_unseenAcc{:.2f}_epoch{}.pt".format(
                            self.args.name, valseen_acc, val_unseen_acc, epoch
                        ),
                    )
                    state = convert_model_to_state(best_model, args, rnn_args, cnn_args)
                    torch.save(state, save_path)

                if valseen_acc > best_tune_acc:
                    best_tune_acc = valseen_acc
                    print("[Tune]: Best valSeen accuracy:", best_tune_acc)
                if val_unseen_acc > best_unseen_acc:
                    best_unseen_acc = val_unseen_acc
                    print("[Tune]: Best valUNseen accuracy:", best_unseen_acc)
                patience = 0
            else:
                patience += 1
                # if patience reachs threshold end the training
                if patience >= self.args.early_stopping:
                    break
            print("Patience:", patience)

        print(f"Best model saved at: {save_path}")

    def load_data(self):
        self.loader = Loader(
            mesh2meters_file=self.args.mesh2meters,
            data_dir=self.args.data_dir,
            image_dir=self.args.image_dir,
        )
        if self.args.train and self.args.increase_train:
            self.loader.build_dataset_extra_train(
                file="train_data.json", args=self.args
            )
        else:
            self.loader.build_dataset(file="train_data.json", args=self.args)
        self.loader.build_dataset(file="valSeen_data.json", args=self.args)
        self.loader.build_dataset(file="valUnseen_data.json", args=self.args)
        self.train_iterator = DataLoader(
            self.loader.datasets["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
        )
        self.valseen_iterator = DataLoader(
            self.loader.datasets["valSeen"],
            batch_size=self.args.batch_size,
            shuffle=False,
        )

        self.val_unseen_iterator = DataLoader(
            self.loader.datasets["valUnseen"],
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        self.scan_graphs["train"] = self.loader.datasets["train"].scan_graphs
        self.scan_graphs["valSeen"] = self.loader.datasets["valSeen"].scan_graphs
        self.scan_graphs["valUnseen"] = self.loader.datasets["valUnseen"].scan_graphs

        if self.args.evaluate:
            self.loader.build_dataset(file="test_data.json", args=self.args)
            self.test_iterator = DataLoader(
                self.loader.datasets["test"],
                batch_size=self.args.batch_size,
                shuffle=False,
            )
            self.scan_graphs["test"] = self.loader.datasets["test"].scan_graphs

    def run(self):
        if self.args.train:
            self.load_data()
            self.run_train()

        elif self.args.evaluate:
            self.run_test()


if __name__ == "__main__":
    args = cfg.parse_args()
    agent = LingUNetAgent(args)
    print(args.data_dir)
    agent.run()