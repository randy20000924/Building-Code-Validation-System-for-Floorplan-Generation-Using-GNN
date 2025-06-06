import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import geopandas as gpd
import pickle as pickle
from collections.abc import Iterable 
from tqdm import tqdm
import os
import shapely
from shapely import affinity, MultiPolygon, GeometryCollection, Polygon, ops, LineString, unary_union, intersection_all
import geopandas as gpd
from pandas.core.series import Series
import networkx as nx
import distinctipy
import random
from torch_geometric.data import Data
import torch
data_url = "../Data/train.pickle"


try:
    df = pickle.load(open(data_url, 'rb'))
except TypeError:
    try:
        df = pd.read_pickle(data_url)
    except Exception as e:
        print(f"Loading failed: {e}")

for col in df.columns:
    df[col] = df[col].apply(lambda x: x if x and len(x.geoms) else None)

df = df.rename(columns={
    'interior_area': 'inner',
    'exterior_wall_2': 'outer_wall',
    'interior_wall': 'inner_wall',
    'second': 'second_room'
})
geoms_columns = ['inner', 'living', 'master', 'kitchen', 'bathroom', 'dining', 'child', 'study',
                   'second_room', 'guest', 'balcony', 'storage', 'wall-in',
                    'outer_wall', 'front', 'inner_wall', 'interior',
                   'front_door', 'outer_wall', 'entrance']
only_rooms =  ['living', 'master', 'kitchen', 'bathroom', 'dining', 'child', 'study',
                   'second_room', 'guest', 'balcony', 'storage', 'entrance']
needed_columns = only_rooms + ['front_door', 'interior']
df.count()
MIN_AREA_PER_ROOM = {
    "living": 12.0,
    "kitchen": 6.0,
    "bathroom": 4.0,
    "master": 10.0,
    "child": 8.0,
    "study": 6.0,
    "dining": 8.0,
    "guest": 9.0,
    "balcony": 3.0,
    "storage": 2.0,
    "entrance": 3.0,
    "second_room": 9.0
}
def get_min_area_dict(row):
    return {room: MIN_AREA_PER_ROOM.get(room, 0.0) for room in only_rooms if isinstance(row.get(room), (Polygon, MultiPolygon))}
df["min_area_dict"] = df.apply(get_min_area_dict, axis=1)


df["min_area_dict"].iloc[0]


def plot_colors_dict(colors):
    """ Plot a dict of rooms colors
    Parameters
    ----------
    colors: dict
        The dict of rooms names as keys and their colors as values
    """
    n = len(colors)
    ncols = 4
    nrows = n // ncols + 1
    fig, ax = plt.subplots(figsize=(9, 2))
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols
    for i, name in enumerate(colors):
        col = i % ncols
        row = i // ncols
        y = Y - (row * h) - h
        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)
        ax.text(xi_text, y, name, fontsize=(h * 0.8),
                horizontalalignment='left',
                verticalalignment='center')
        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  color=tuple([*colors[name]/255, 1]), linewidth=(h * 0.6))
    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)
    plt.show()
def imshow(img, fig_size=(500, 500)):
    """ Plot an image
    Parameters
    ----------
    img: ndarray (OpenCV image) or PIL.Image
        The image to be plotted
    fig_size: tuple
        The figure size  
    """
    try:
        img = Image.fromarray(img)
    except:
        try:
            img = img.astype(np.uint8)
        except:
            ...
    display(img.resize(fig_size))

random.seed(2000)
N = len(geoms_columns)
colors = (np.array(distinctipy.get_colors(N)) * 255).astype(np.uint8)
room_color = {room_name: colors[i] for i, room_name in enumerate(geoms_columns)}

def get_mask(poly, shape):
    """ Return image contains multiploygon as a numpy array mask
    Parameters
    ----------
    poly: Polygon or MultiPolygon or Iterable[Polygon or MultiPolygon]
        The Polygon/s to get mask for
    shape: tuple
        The shape of the canvas to draw polygon/s on
    Returns
    -------
    ndarray
        Mask array of the input polygon/s
    """
    img = np.zeros(shape, dtype=np.uint8)
    if isinstance(poly, Polygon):
        img = cv2.drawContours(img, np.int32([poly.exterior.coords]), -1, 255, -1)
    elif isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            img = cv2.drawContours(img, np.int32([p.exterior.coords]), -1, 255, -1)
    elif isinstance(poly, Series):
        polys = [p for p in poly.tolist() if p]
        img = get_mask(polys, shape)
    elif isinstance(poly, Iterable):
        for p in poly:
            img = (img != 0) | (get_mask(p, shape) != 0)
        img = img.astype(np.uint8) * 255
    return img.astype(np.uint8)
def plot_plan_colored(index=None, idf=df, row=None, shape=(255, 255), 
                      return_img=False, points=None, radius=5, 
                      rooms_columns=geoms_columns):
    """ Plot plan given index or Series
    Parameters
    ----------
    index: int, optional
        The index of the plan in the Dataframe
    idf: Dataframe, optional
        The Dataframe of floor plans to index from
    row: Series, optional
        The row Series containing the plan
    shape: tuple:
        The size of the canvas to draw on
    points: list, optional:
        List of points to draw on the image
    radius: list, optional:
        Radius of points (if exists)
    rooms_columns: list, optional
        Names of rooms columns to be plotted
    """
    img  = np.zeros((*shape, 3), dtype=np.uint8)
    if row is None:
        if index is not None:
            row = idf.iloc[index]
        else:
            return
    for i, room_name in enumerate(rooms_columns):
        color = room_color[room_name]
        room_geo = row[room_name]
        if room_geo is not None:
            mask = get_mask(room_geo, shape)
            img[np.where(mask > 0)] = color
    if points:
        for p in points:
            cv2.circle(img, (int(p[0]), int(p[1])), radius, (255, 255, 255), thickness=-1)
    imshow(img)
    if return_img:
        return img
def plot_plan_list_colored(plan_list, shape=(255, 255), 
                      return_img=False, points=None, radius=5):
    """ Plot plan given plan list of lists: [['living', poly_livint], ['living', poly2_living]]
    Parameters
    ----------
    plan_list: list
        The index of the plan in the Dataframe
    shape: tuple:
        The size of the canvas to draw on
    points: list, optional:
        List of points to draw on the image
    radius: list, optional:
        Radius of points (if exists)
    """
    img  = np.zeros((*shape, 3), dtype=np.uint8)
    for i, (room_name, polygon) in enumerate(plan_list):
        color = room_color[room_name]
        room_geo = polygon
        if room_geo is not None:
            mask = get_mask(room_geo, shape)
            img[np.where(mask > 0)] = color
    if points:
        for p in points:
            cv2.circle(img, (int(p[0]), int(p[1])), radius, (255, 255, 255), thickness=-1)
    imshow(img)
    if return_img:
        return img
def get_real_area(mpoly, base_img_area=256*256, orig_area=18*18):
    """ Get real area in meters for a MultiPolygon: 
    Parameters
    ----------
    mpoly: MultiPolygon
        The multipolygon to get area of.
    base_img_area: int
        The area of the image this Polygon extracted from
    orig_area: int
        The scale of the base image in square meters
    """
    return orig_area * mpoly.area / base_img_area


idx = 3977
interior = df.iloc[idx]['inner']
wall = df.iloc[idx]['outer_wall']
interior_and_wall = interior.union(wall)
get_real_area(interior_and_wall)

plot_plan_colored(900)

df[df['front_door'].isna()]

df.shape[0]

df = df.dropna(subset=['front_door'])
df = df.reset_index()
df.shape[0]

def get_rooms_polygons(cell):
    ''' apply function for dataframe to convert MultiPolygons to a list of Polygons'''
    return list(cell.geoms) if cell is not None else []
df_rooms_polys = df[needed_columns].applymap(get_rooms_polygons)

def get_rooms_polygons_dict(row):
    ''' apply function for dataframe toa get dict of rooms and their polygons'''
    return row.to_dict()
df_rooms_polys_dict = df_rooms_polys.apply(get_rooms_polygons_dict, axis=1)

embedings = {}
for i, type_ in enumerate(df_rooms_polys_dict[0].keys()):
    embedings[type_] = i
embedings

w_list = []
h_list = []
import json
with open("C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/Creating_Dataset/min_area_stats.json", "r") as f:
    stats = json.load(f)
def createGraph(Graph_index, living_to_all=False, all_conected=False):
    """
    Generating a graph for a specific floor plan
    Input: 
        Graph_index: Index of the floo plan.
    Output:
        G: a networkx graph.
    """
    floor_plan = df_rooms_polys_dict[Graph_index]
    G = nx.Graph()
    n = len(floor_plan['interior'])
    summation = 0 
    if n != 0: # if there are inner doors
        for i in range(n):
            x1, y1, x2, y2 = floor_plan['interior'][i].bounds
            res1, res2 = x2 - x1, y2 - y1
            summation += min(res1, res2)
    else: # if there is no inner doors, take the outer.
        n = 1
        x1, y1, x2, y2 = floor_plan['front_door'][0].bounds
        res1, res2 = x2 - x1, y2 - y1
        summation += min(res1, res2)
    threshold = summation / n
    for type_, mPoly in floor_plan.items():
        if (len(mPoly) == 0) or (type_ in ['interior', 'front_door', 'balcony', 'storage', 'entrance']):
            continue
        else:
            for i, poly1 in enumerate(mPoly):
                currentNodeName = f"{type_}_{i}"
                center_x = poly1.centroid.coords[0][0]
                center_y = poly1.centroid.coords[0][1]
                rec_w = poly1.bounds[2] - poly1.bounds[0]
                rec_h = poly1.bounds[3] - poly1.bounds[1]
                w_list.append(rec_w)
                h_list.append(rec_h)
                raw_min_area = df.loc[Graph_index, "min_area_dict"].get(type_, 6.0)
                min_area = (raw_min_area - stats["mean"]) / stats["std"]
                G.add_node(currentNodeName,
                    roomType_name = type_,
                    roomType_embd = embedings[type_],
                    actualCentroid_x = center_x,
                    actualCentroid_y = center_y,
                    rec_w = rec_w,
                    rec_h = rec_h,
                    roomSize = poly1.area / df['inner'][Graph_index].area,
                    min_area = min_area
                )
                if (not living_to_all) and (not all_conected) :
                    for type__, mPoly_ in floor_plan.items():
                        if (len(mPoly_) == 0) or (type__ in ['interior', 'front_door', 'balcony', 'storage', 'entrance']):
                            continue
                        else:
                            for j, poly2 in enumerate(mPoly_):
                                if poly2 == poly1:
                                    continue
                                else:
                                    p1 = poly1.buffer(threshold)
                                    p2 = poly2.buffer(threshold)
                                    if p1.intersects(p2):
                                        adjNodeName = f'{type__}_{j}'
                                        dis = poly1.centroid.distance(poly2.centroid)
                                        G.add_edge(currentNodeName, adjNodeName, distance=round(dis, 3))
    if living_to_all: 
        living_cen = shapely.Point(G.nodes['living_0']['actualCentroid_x'], G.nodes['living_0']['actualCentroid_y'])
        for node in G.nodes():
                if G.nodes[node]['roomType_name'] != 'living':
                    point = shapely.Point(G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y'])
                    dis = living_cen.distance(point)
                    G.add_edge('living_0', node, distance=round(dis, 3))
    if all_conected: 
        for node in G.nodes():
            current_node_centeroid = shapely.Point(G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y'])
            for other_node in G.nodes():
                if other_node != node: # for all other rooms
                    other_node_centeroid = shapely.Point(G.nodes[other_node]['actualCentroid_x'], G.nodes[other_node]['actualCentroid_y'])
                    dis = current_node_centeroid.distance(other_node_centeroid)
                    G.add_edge(node, other_node, distance=round(dis, 3))
    return G

def draw_graph_nodes(G, living_to_all=False):
    pos = {node: (G.nodes[node]['actualCentroid_x'], -G.nodes[node]['actualCentroid_y']) for node in G.nodes}
    scales = [G.nodes[node]['roomSize'] * 10000 for node in G] 
    color_map = [room_color[G.nodes[node]['roomType_name']]/255 for node in G]
    edge_labels = nx.get_edge_attributes(G, 'distance')
    nx.draw_networkx_nodes(G, pos=pos, node_size=scales, node_color=color_map);
    nx.draw_networkx_edges(G, pos=pos, edge_color='b');
    nx.draw_networkx_labels(G, pos=pos, font_size=8);
    if living_to_all:
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.xlim(-10, 266)
    plt.ylim(-266, 10)

def draw_graph_boundary(G):
    pos = {node: (G.nodes[node]['centroid'][0], -G.nodes[node]['centroid'][1])  for node in G.nodes}
    door_color = '#90EE90'
    other_nodes_color = '#0A2A5B'
    color_map = [door_color if G.nodes[node]['type'] == 1 else other_nodes_color for node in G.nodes]
    nx.draw_networkx_nodes(G, pos=pos, node_size=150, node_color=color_map);
    nx.draw_networkx_edges(G, pos=pos)
    plt.xlim(-10, 266)
    plt.ylim(-266, 10)

coords = df.inner[2].geoms[0].exterior.coords[:]
points = []
for p in coords:
    points.append(shapely.Point(p))
graph = nx.Graph()
graph.add_node(0, type=0, centroid=coords[0])
print('0 is Done', '\n', '='*50)
current = 0
name = 1
for i in range(1, len(coords)):
    print(f'num_of_nodes: {len(graph)}')
    dis = points[i].distance(points[current])
    print(i, current, '--> ', dis )
    if dis >= 5:
        print(i, 'Done')
        graph.add_node(name, type=0, centroid=coords[i])
        current = i
        name += 1
    else:
        print(i, 'Not-Done')
    print('='*50)
nodes_names = list(graph.nodes)
print(graph.nodes[nodes_names[0]])
print(f'Number of nodes now: {len(nodes_names)}')
first_node = shapely.Point(graph.nodes[nodes_names[0]]['centroid'])
last_node  = shapely.Point(graph.nodes[nodes_names[-1]]['centroid'])
if first_node.distance(last_node) <= 5:
    graph.remove_node(nodes_names[-1])
    nodes_names = list(graph.nodes)
    print(f'Num of nodes after removing: {len(nodes_names)}')
points = []
for node in graph:
    points.append(shapely.Point(graph.nodes[node]['centroid']))
for i in range(len(nodes_names)-1):
    dis = points[i].distance(points[i+1])
    graph.add_edge(nodes_names[i],nodes_names[i+1], distance=dis)
dis = points[nodes_names[0]].distance(points[nodes_names[-1]])
graph.add_edge(nodes_names[0], nodes_names[-1], distance=dis)

def adding_door(boundary_graph, index, points):
    door = df['front_door'][index]
    nearest_edge = None
    nearest_dist = float('inf')
    dx = door.bounds[2] - door.bounds[0]
    dy = door.bounds[3] - door.bounds[1]
    door_oriantation_horizontal = dx > dy
    for edge in boundary_graph.edges():
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        line = shapely.LineString([p1, p2])
        p1x, p1y = p1.x, p1.y
        p2x, p2y = p2.x, p2.y
        dx = abs(p2x - p1x)
        dy = abs(p2y - p1y)
        line_oriantation_horizontal = dx > dy
        if door_oriantation_horizontal == line_oriantation_horizontal:
            dist = door.distance(line)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_edge = edge
    boundary_graph.remove_edge(*nearest_edge)
    door_ind = len(boundary_graph)
    door_centroid = door.geoms[0].centroid
    boundary_graph.add_node(door_ind, type=1, centroid=(door_centroid.x, door_centroid.y))
    dist = door_centroid.distance(shapely.Point(boundary_graph.nodes[nearest_edge[0]]['centroid']))
    boundary_graph.add_edge(nearest_edge[0], door_ind, distance=dist)
    dist = door_centroid.distance(shapely.Point(boundary_graph.nodes[nearest_edge[1]]['centroid']))
    boundary_graph.add_edge(nearest_edge[1], door_ind, distance=dist)
    return boundary_graph

def Handling_dubplicated_nodes(graph_index):
    coords = df.inner[graph_index].geoms[0].exterior.coords[:]
    points = []
    for p in coords:
        points.append(shapely.Point(p))
    graph = nx.Graph()
    graph.add_node(0, type=0, centroid=coords[0])
    current = 0
    name = 1
    for i in range(1, len(coords)):
        dis = points[i].distance(points[current])
        if dis >= 5:
            graph.add_node(name, type=0, centroid=coords[i])
            current = i
            name += 1
    nodes_names = list(graph.nodes)
    first_node = shapely.Point(graph.nodes[nodes_names[0]]['centroid'])
    last_node  = shapely.Point(graph.nodes[nodes_names[-1]]['centroid'])
    if first_node.distance(last_node) <= 5:
        graph.remove_node(nodes_names[-1])
        nodes_names = list(graph.nodes)
    points_of_current_graph = []
    for node in graph:
        points_of_current_graph.append(shapely.Point(graph.nodes[node]['centroid']))
    for i in range(len(nodes_names)-1):
        dis = points_of_current_graph[i].distance(points_of_current_graph[i+1])
        graph.add_edge(nodes_names[i],nodes_names[i+1], distance=dis)
    dis = points_of_current_graph[nodes_names[0]].distance(points_of_current_graph[nodes_names[-1]])
    graph.add_edge(nodes_names[0], nodes_names[-1], distance=dis)
    graph = adding_door(graph, graph_index, points_of_current_graph)
    return graph

Graph_index = 1911
G = createGraph(Graph_index)
b = Handling_dubplicated_nodes(Graph_index)
draw_graph_nodes(G); 
plot_plan_colored(Graph_index, df);
draw_graph_boundary(b);

list(G.nodes(data=True))[0]


G_living_to_all = createGraph(Graph_index, living_to_all=True)
draw_graph_nodes(G_living_to_all, living_to_all=True); 
draw_graph_boundary(b);

G_all_conected = createGraph(Graph_index, all_conected=True)
b = Handling_dubplicated_nodes(Graph_index)
draw_graph_nodes(G_all_conected); 
draw_graph_boundary(b);

Graphs_real = []
Graphs_living_to_all = []
Graphs_all_conected = []
boundaries = []
for idx in tqdm(range(len(df_rooms_polys_dict))):
    G_real = createGraph(idx) 
    G_living_to_all = createGraph(idx, living_to_all=True)
    G_all_conected = createGraph(idx, all_conected=True)
    b = Handling_dubplicated_nodes(idx)
    Graphs_real.append(G_real)
    Graphs_living_to_all.append(G_living_to_all)
    Graphs_all_conected.append(G_all_conected)
    boundaries.append(b)


import pickle
import json
min_areas = []
for G in Graphs_living_to_all:
    for _, attr in G.nodes(data=True):
        if "min_area" in attr:
            min_areas.append(attr["min_area"])
min_area_mean = sum(min_areas) / len(min_areas)
min_area_std = (sum((x - min_area_mean) ** 2 for x in min_areas) / len(min_areas)) ** 0.5
print(f"min_area mean: {min_area_mean:.3f}")
print(f"min_area std: {min_area_std:.3f}")
for G in Graphs_real + Graphs_living_to_all + Graphs_all_conected:
    for _, attr in G.nodes(data=True):
        if "min_area" in attr:
            attr["min_area"] = (attr["min_area"] - min_area_mean) / min_area_std
print("min_area normalized")
with open("./graphs/Graphs_living_to_all_normalized.pkl", "wb") as f:
    pickle.dump(Graphs_living_to_all, f)
print("normalized Graphs saved")


