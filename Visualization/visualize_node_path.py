import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import cv2
import imageio
import os
from scipy import ndimage

node2pixel = json.load(open("../data/floorplans/allScans_Node2pix.json"))
connectivity_dir = "../data/connectivity/"
new_data_dir = "../data/floorplans/"


def draw_graph(scanId, floor, path):
    # loads the top down map
    im = np.array(
        cv2.imread(
            new_data_dir
            + "floor_"
            + str(floor)
            + "/"
            + scanId
            + "_"
            + str(floor)
            + ".png"
        ),
        dtype=np.uint8,
    )

    # draws connections between nodes
    for n in range(len(path) - 1):
        point1 = node2pixel[scanId][path[n]][0]
        point2 = node2pixel[scanId][path[n + 1]][0]
        im = cv2.line(im, tuple(point1), tuple(point2), (0, 0, 0), 3)

    # draws nodes on image
    for node in path:
        position = tuple(node2pixel[scanId][node][0])
        im = cv2.circle(im, position, 10, (255, 0, 0), -1)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(im)
    plt.savefig("graph.png", bbox_inches="tight")


if __name__ == "__main__":

    # need to specify scan and floor, you can optionally load a specific node
    scanId = "17DRP5sb8fy"
    floor = 0
    path = [
        "db145474a5fa476d95c2cc7f09e7c83a",
        "5b9b2794954e4694a45fc424a8643081",
        "51857544c192476faebf212acb1b3d90",
        "e34dcf54d26a4a95869cc8a0c01cd2be",
        "0f37bd0737e349de9d536263a4bdd60d",
    ]

    draw_graph(scanId, floor, path)
