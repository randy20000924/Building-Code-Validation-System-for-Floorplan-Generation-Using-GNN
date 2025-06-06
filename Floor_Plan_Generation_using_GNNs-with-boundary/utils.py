# for data wrangling
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import distinctipy
from collections import defaultdict

from torch_geometric.utils import from_networkx
import torch

# for saving and loading the images
import os
import uuid
import time


# from shapely import Point, MultiPolygon, GeometryCollection, Polygon, ops, LineString, unary_union, intersection_all
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon, Point, LineString, box
from shapely.ops import unary_union
import shapely.affinity as aff
from shapely.wkt import loads
import geopandas as gpd

room_embeddings = {
    'living': 0,
    'room': 1,
    'kitchen': 2,
    'bathroom': 3,
}

poly_types = list(room_embeddings.keys())
N = len(poly_types)
colors = (np.array(distinctipy.get_colors(N)) * 255).astype(np.uint8)
room_color = {room_name: colors[i] for i, room_name in enumerate(poly_types)}

def Handling_dubplicated_nodes(boundary, door):
    coords = boundary.exterior.coords[:]
        
    points = []
    for p in coords:
        points.append(Point(p))

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
    first_node = Point(graph.nodes[nodes_names[0]]['centroid'])
    last_node  = Point(graph.nodes[nodes_names[-1]]['centroid'])
    if first_node.distance(last_node) <= 5:
        graph.remove_node(nodes_names[-1])
        nodes_names = list(graph.nodes)
        
    points_of_current_graph = []
    for node in graph:
        points_of_current_graph.append(Point(graph.nodes[node]['centroid']))

    for i in range(len(nodes_names)-1):
        dis = points_of_current_graph[i].distance(points_of_current_graph[i+1])
        graph.add_edge(nodes_names[i],nodes_names[i+1], distance=dis)

    dis = points_of_current_graph[nodes_names[0]].distance(points_of_current_graph[nodes_names[-1]])

    graph.add_edge(nodes_names[0], nodes_names[-1], distance=dis)
    
    graph = adding_door(graph, door, points_of_current_graph)
    
    return graph

def adding_door(boundary_graph, door, points):
    nearest_edge = None
    nearest_dist = float('inf')
    
    dx = door.bounds[2] - door.bounds[0]
    dy = door.bounds[3] - door.bounds[1]
    door_oriantation_horizontal = dx > dy

    for edge in boundary_graph.edges():
        p1 = points[edge[0]]
        p2 = points[edge[1]]

        line = LineString([p1, p2])

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
    door_centroid = door.centroid
    boundary_graph.add_node(door_ind, type=1, centroid=(door_centroid.x, door_centroid.y))

    dist = door_centroid.distance(Point(boundary_graph.nodes[nearest_edge[0]]['centroid']))
    boundary_graph.add_edge(nearest_edge[0], door_ind, distance=dist)

    dist = door_centroid.distance(Point(boundary_graph.nodes[nearest_edge[1]]['centroid']))
    boundary_graph.add_edge(nearest_edge[1], door_ind, distance=dist)
    
    return boundary_graph

def centroids_to_graph(floor_plan, living_to_all=False, all_conected=False):
    import json
    G = nx.Graph()

    with open("C:/Users/jensonyu/Documents/ENGR project/floor_plan_verification_project/data/room_min_sizes.json") as f:
        room_min_sizes = json.load(f)
    
    for type_, list_of_centroids in floor_plan.items():
        for i, centroid in enumerate(list_of_centroids):
            currentNodeName = f'{type_}_{i}'
            type_map = {
                'room': 'Bedroom',
                'kitchen': 'Kitchen',
                'bathroom': 'Bathroom',
            }

            json_key = type_map.get(type_, type_.capitalize())
            min_area_m2 = room_min_sizes.get(json_key, 6.0)
            min_area = min_area_m2 * 2.6  # 1m² = 4 pixel²

            G.add_node(currentNodeName,
                roomType_name = type_,
                roomType_embd = room_embeddings[type_],
                actualCentroid_x = centroid[0],
                actualCentroid_y = centroid[1],
                min_area = min_area
            )
                                        
    if living_to_all: 
        living_cen = Point(G.nodes['living_0']['actualCentroid_x'], G.nodes['living_0']['actualCentroid_y'])
        for node in G.nodes():
                if G.nodes[node]['roomType_name'] != 'living':
                    point = Point(G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y'])
                    dis = living_cen.distance(point)
                    G.add_edge('living_0', node, distance=round(dis, 3))
                    
    if all_conected: 
        for node in G.nodes():
            current_node_centeroid = Point(G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y'])

            for other_node in G.nodes():
                if other_node != node:
                    other_node_centeroid = Point(G.nodes[other_node]['actualCentroid_x'], G.nodes[other_node]['actualCentroid_y'])
                    dis = current_node_centeroid.distance(other_node_centeroid)
                    G.add_edge(node, other_node, distance=round(dis, 3))

    return G


def boundary_to_image(boundary_wkt, front_door_wkt):
    boundary = shapely.wkt.loads(boundary_wkt)
    front_door = shapely.wkt.loads(front_door_wkt)
    
    boundary = scale(boundary)
    front_door = scale(front_door)
    
    plt.figure(figsize=(5, 5))
    gpd.GeoSeries([boundary, front_door]).plot(cmap='tab10');
    
    path = os.getcwd() + "/Outputs/boundary.png"
    plt.savefig(path)
    plt.close()
    
    return path
    
def get_user_inputs_as_image(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids):
    boundary = shapely.wkt.loads(boundary_wkt)
    front_door = shapely.wkt.loads(front_door_wkt)
    
    boundary = scale(boundary)
    front_door = scale(front_door)
    room_centroids = [scale(x) for x in room_centroids]
    bathroom_centroids = [scale(x) for x in bathroom_centroids]
    kitchen_centroids = [scale(x) for x in kitchen_centroids]
    
    polys = defaultdict(list)

    for center in room_centroids:
        polys['room'].append(center)

    for center in bathroom_centroids:
        polys['bathroom'].append(center)

    for center in kitchen_centroids:
        polys['kitchen'].append(center)

    Input_format = []
    Input_format.append(boundary)
    Input_format.append(front_door)

    for _, poly_list in polys.items():
        Input_format.append(unary_union(poly_list))

    Input_format = gpd.GeoSeries(Input_format)
    Input_format.plot(cmap='twilight', alpha=0.8, linewidth=0.8, edgecolor='black');
    
    path = os.getcwd() + '/Outputs/user_inputs.png'
    plt.savefig(path)
    plt.close()
    
    return path

def draw_graph(G):
    pos = {node: (G.nodes[node]['actualCentroid_x'], G.nodes[node]['actualCentroid_y']) for node in G.nodes}
    
    colormap = [room_color[G.nodes[node]['roomType_name']]/255 for node in G]
    
    nx.draw(G, pos=pos, node_color=colormap, with_labels=True, font_size=12)
    
    
def draw_graph_boundary(G):
    pos = {node: (G.nodes[node]['centroid'][0], G.nodes[node]['centroid'][1])  for node in G.nodes}
    
    door_color = '#90EE90'
    other_nodes_color = '#0A2A5B'
    color_map = [door_color if G.nodes[node]['type'] == 1 else other_nodes_color for node in G.nodes]
    
    nx.draw(G, pos=pos, with_labels=True, node_color=color_map, font_color='w', font_size=12)
    
def draw_both_graphs(boundary_graph, entire_graph):
    plt.figure()
    
    draw_graph_boundary(boundary_graph)
        
    draw_graph(entire_graph)
    
    path = os.getcwd() + '/Outputs/both_graphs.png'
    plt.savefig(path)
    plt.close()  # Close the figure to free up resources
    
    return path
    
def scale(x):
    if isinstance(x, tuple):
        x = Point(*x)
        
    return aff.scale(x, xfact=1, yfact=-1, origin=(128, 128))

class FloorPlan_multipolygon():
    def __init__(self, graph, prediction):
        self.graph       = graph
        self.prediction  = prediction
        
    def get_room_data(self, room_index):
        centroid = (self.graph.x[room_index][-2].item(), self.graph.x[room_index][-1].item())
        category = torch.argmax(self.graph.x[:, :7], axis=1)[room_index].item()
        w_pre, h_pre = self.get_predictions(room_index)
            

        data = {
            'centroid': centroid,
            'predic_w': w_pre,
            'predic_h': h_pre,
            'category': category
        }
        return data
    
    def create_box(self, room_data):
        centroid = room_data['centroid']
        half_w = room_data['predic_w'] / 2
        half_h = room_data['predic_h'] / 2

        x1 = centroid[0] - half_w
        x2 = centroid[0] + half_w
        y1 = centroid[1] - half_h
        y2 = centroid[1] + half_h

        return box(x1, y1, x2, y2)


    def get_multipoly(self, boundary=False, door=False):
        num_of_rooms = self.graph.x.shape[0]
        rooms_by_category = defaultdict(list)
        
        for index in range(num_of_rooms):
            room_data = self.get_room_data(index)
            box = self.create_box(room_data)
            box = box.intersection(boundary.buffer(-3, cap_style=3, join_style=2))

            if box.area < 10 or not box.is_valid:
                continue
                
            room_category = room_data['category']
            rooms_by_category[room_category].append({
                'box': box,
                'data': room_data,
                'area': box.area
            })

        for category in rooms_by_category:
            rooms_by_category[category].sort(key=lambda x: x['area'], reverse=True)
        
        final_polygons = [boundary]
        all_processed_rooms = []

        for category in [0, 1, 2, 3]:  # living, bedroom, kitchen, bathroom
            if category not in rooms_by_category:
                continue

            for room in rooms_by_category[category]:
                room_poly = room['box']
                overlap_flag = False

                for existing_room in all_processed_rooms:
                    intersection = room_poly.intersection(existing_room)
                    if intersection.area > 0.1 * room_poly.area:
                        room_poly = room_poly.difference(intersection.buffer(1))
                        if isinstance(room_poly, MultiPolygon):
                            largest_piece = max(room_poly.geoms, key=lambda x: x.area)
                            room_poly = largest_piece if largest_piece.area > 0 else room_poly

                # Final check after difference
                if room_poly.area > 10 and room_poly.is_valid:
                    all_processed_rooms.append(room_poly)
                    final_polygons.append(room_poly)

        if door:
            final_polygons.append(door)
        
        return gpd.GeoSeries(final_polygons)

    
    def get_predictions(self, room_index):
        w_predicted = self.prediction[room_index, 0]
        h_predicted = self.prediction[room_index, 1]
        
        return w_predicted, h_predicted
