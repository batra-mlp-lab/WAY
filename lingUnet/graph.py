import json
import networkx as nx
import math
import numpy as np


def snap_to_grid(G, node2pix, sn, pred_coord, conversion, level, true_viewpoint=None):
    min_dist = math.inf
    best_nodes = []
    best_node = ""
    for node in node2pix[sn].keys():
        if node2pix[sn][node][2] != int(level) or node not in G:
            continue
        target_coord = [node2pix[sn][node][0][1], node2pix[sn][node][0][0]]
        dist = np.sqrt(
            (target_coord[0] - pred_coord[0]) ** 2
            + (target_coord[1] - pred_coord[1]) ** 2
        ) / (conversion)
        if dist.item() < min_dist:
            best_node = node
            min_dist = dist.item()
        if dist < 1:
            best_nodes.append(node)

    if true_viewpoint != None:
        min_dist = math.inf
        for b in best_nodes:
            dist = get_geo_dist(G, b, true_viewpoint)
            if dist < min_dist:
                best_node = b
                min_dist = dist
    return best_node


def distance(pose1, pose2):
    """ Euclidean distance between two graph poses """
    return (
        (pose1["pose"][3] - pose2["pose"][3]) ** 2
        + (pose1["pose"][7] - pose2["pose"][7]) ** 2
        + (pose1["pose"][11] - pose2["pose"][11]) ** 2
    ) ** 0.5


def open_graph(connectDir, scan_id):
    """ Build a graph from a connectivity json file """
    infile = "%s%s_connectivity.json" % (connectDir, scan_id)
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


def get_geo_dist(D, n1, n2):
    return nx.dijkstra_path_length(D, n1, n2)
