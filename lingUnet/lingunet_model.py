import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

import copy

from loader import Loader
from language_model import RNN


class LinearProjectionLayers(nn.Module):
    def __init__(
        self, image_channels, linear_hidden_size, rnn_hidden_size, num_hidden_layers
    ):
        super(LinearProjectionLayers, self).__init__()

        if num_hidden_layers == 0:
            # map pixel feature vector directly to score without activation
            self.out_layers = nn.Linear(image_channels + rnn_hidden_size, 1, bias=False)
        else:
            linear_hidden_layers = []
            for _ in range(num_hidden_layers):
                linear_hidden_layers += [
                    nn.Linear(linear_hidden_size, linear_hidden_size),
                    nn.ReLU(),
                ]

            self.out_layers = nn.Sequential(
                nn.Conv2d(
                    image_channels + rnn_hidden_size,
                    linear_hidden_size,
                    kernel_size=1,
                    padding=0,
                    stride=1,
                ),
                nn.ReLU(),
                nn.Conv2d(linear_hidden_size, 1, kernel_size=1, padding=0, stride=1),
            )

    def forward(self, x):
        return self.out_layers(x)


def clones(module, N):
    """Produce N identical layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LingUNet(nn.Module):
    def __init__(self, rnn_args, cnn_args, args):
        super(LingUNet, self).__init__()
        self.cnn_args = cnn_args
        self.rnn_args = rnn_args

        self.m = args.num_lingunet_layers
        self.image_channels = args.linear_hidden_size
        self.blind_lang = args.blind_lang
        self.blind_vis = args.blind_vis
        self.freeze_resnet = args.freeze_resnet
        self.res_connect = args.res_connect
        self.device = args.device
        self.avgpool = args.avgpool

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-4]
        self.resnet = nn.Sequential(*modules)
        if self.freeze_resnet:
            for p in self.resnet.parameters():
                p.requires_grad = False

        if not args.bidirectional:
            self.rnn_hidden_size = args.rnn_hidden_size
        else:
            self.rnn_hidden_size = args.rnn_hidden_size * 2
        assert self.rnn_hidden_size % self.m == 0

        self.rnn = RNN(
            rnn_args["input_size"],
            args.embed_size,
            args.rnn_hidden_size,
            args.num_rnn_layers,
            args.embed_dropout,
            args.bidirectional,
            args.embedding_type,
            args.embedding_dir,
        ).to(args.device)

        sliced_text_vector_size = self.rnn_hidden_size // self.m
        flattened_conv_filter_size = 1 * 1 * self.image_channels * self.image_channels
        self.text2convs = clones(
            nn.Linear(sliced_text_vector_size, flattened_conv_filter_size), self.m
        )

        self.conv_layers = nn.ModuleList([])
        for i in range(self.m):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.image_channels
                        if i == 0
                        else self.image_channels,
                        out_channels=self.image_channels,
                        kernel_size=cnn_args["kernel_size"],
                        padding=cnn_args["padding"],
                        stride=1,
                    ),
                    nn.BatchNorm2d(self.image_channels),
                    nn.ReLU(True),
                )
            )

        # create deconv layers with appropriate paddings
        self.deconv_layers = nn.ModuleList([])
        for i in range(self.m):
            in_channels = self.image_channels if i == 0 else self.image_channels * 2
            out_channels = self.image_channels
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=cnn_args["kernel_size"],
                        padding=cnn_args["padding"],
                        stride=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )

        self.conv_dropout = nn.Dropout(p=cnn_args["conv_dropout"])
        self.deconv_dropout = nn.Dropout(p=cnn_args["deconv_dropout"])

        self.out_layers = LinearProjectionLayers(
            image_channels=self.image_channels,
            linear_hidden_size=args.linear_hidden_size,
            rnn_hidden_size=0,
            num_hidden_layers=args.num_linear_hidden_layers,
        )

        if self.avgpool:
            self.avg_pool = nn.AvgPool2d(3, 1, padding=1)

    def forward(self, images, texts, seq_lengths):
        if self.blind_lang:
            texts = texts.fill_(0.0)
        if self.blind_vis:
            images = images.fill_(0.0)

        images = self.resnet(images)

        batch_size, image_channels, height, width = images.size()

        text_embed = self.rnn(texts, seq_lengths)
        sliced_size = self.rnn_hidden_size // self.m

        Gs = []
        image_embeds = images

        for i in range(self.m):
            image_embeds = self.conv_dropout(image_embeds)
            image_embeds = self.conv_layers[i](image_embeds)
            text_slice = text_embed[:, i * sliced_size : (i + 1) * sliced_size]

            conv_kernel_shape = (
                batch_size,
                self.image_channels,
                self.image_channels,
                1,
                1,
            )
            text_conv_filters = self.text2convs[i](text_slice).view(conv_kernel_shape)

            orig_size = image_embeds.size()
            image_embeds = image_embeds.view(1, -1, *image_embeds.size()[2:])
            text_conv_filters = text_conv_filters.view(
                -1, *text_conv_filters.size()[2:]
            )
            G = F.conv2d(image_embeds, text_conv_filters, groups=orig_size[0]).view(
                orig_size
            )
            image_embeds = image_embeds.view(orig_size)
            if self.res_connect:
                G = G + image_embeds
                G = F.relu(G)
            Gs.append(G)

        # deconvolution operations, from the bottom up
        H = Gs.pop()
        for i in range(self.m):
            if i == 0:
                H = self.deconv_dropout(H)
                H = self.deconv_layers[i](H)
            else:
                G = Gs.pop()
                concated = torch.cat((H, G), 1)
                H = self.deconv_layers[i](concated)

        out = self.out_layers(H).squeeze(-1)
        if self.avgpool:
            out = self.avg_pool(out)

        out = F.log_softmax(out.view(batch_size, -1), 1).view(batch_size, height, width)
        return out