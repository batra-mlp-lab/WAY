import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import cv2

node2pixel = json.load(open("../data/floorplans/allScans_Node2pix.json"))
connectivity_dir = "../data/connectivity/"
new_data_dir = "../data/floorplans/"


def distance(pose1, pose2):
    """ Euclidean distance between two graph poses """
    return (
        (pose1["pose"][3] - pose2["pose"][3]) ** 2
        + (pose1["pose"][7] - pose2["pose"][7]) ** 2
        + (pose1["pose"][11] - pose2["pose"][11]) ** 2
    ) ** 0.5


def open_graph(scan_id):
    """ Build a graph from a connectivity json file """
    infile = "%s%s_connectivity.json" % (connectivity_dir, scan_id)
    G = nx.Graph()
    with open(infile) as f:
        data = json.load(f)
        for i, item in enumerate(data):
            if item["included"]:
                for j, conn in enumerate(item["unobstructed"]):
                    if conn and data[j]["included"]:
                        assert data[j]["unobstructed"][i], "Graph should be undirected"
                        G.add_edge(
                            item["image_id"],
                            data[j]["image_id"],
                            weight=distance(item, data[j]),
                        )
    return G


def draw_graph(scanId, floor):
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

    nodes = []
    # draws nodes on image
    for viewPointId, value in node2pixel[scanId].items():
        if value[-1] == floor:
            nodes.append(viewPointId)
            im = cv2.circle(im, tuple(value[0]), 10, (255, 0, 0), -1)
    # draws connections between nodes
    g = open_graph(scanId)
    for node in nodes:
        point1 = node2pixel[scanId][node][0]
        for edge in list(g.edges(node)):
            if edge[1] in nodes:
                point2 = node2pixel[scanId][edge[1]][0]
                im = cv2.line(im, tuple(point1), tuple(point2), (255, 0, 0), 3)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.axis("off")
    plt.imshow(im)
    plt.savefig("graph.png", bbox_inches="tight")


if __name__ == "__main__":
    # need to specify scan and floor, you can optionally load a specific node
    scanId = "2t7WUuJeko7"
    floor = 0
    specificNode = None
    draw_graph(scanId, floor)
