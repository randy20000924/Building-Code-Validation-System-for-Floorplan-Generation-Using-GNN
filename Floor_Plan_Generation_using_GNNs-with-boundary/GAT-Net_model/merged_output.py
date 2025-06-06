import pickle
import matplotlib.pyplot as plt
with open(r"C:\Users\jensonyu\Documents\ENGR project\Floor_Plan_Generation_using_GNNs-with-boundary\Creating_Dataset\graphs\Graphs_living_to_all_normalized.pkl", "rb") as f:
    dataset = pickle.load(f)
min_areas = []
for i, G in enumerate(dataset[:50]):  # 檢查前 50 筆
    for node, attrs in G.nodes(data=True):
        if "min_area" in attrs:
            min_areas.append(attrs["min_area"])
print("min_area 數量:", len(min_areas))
print("min_area 範圍: min =", min(min_areas), ", max =", max(min_areas))
print("min_area 平均值:", sum(min_areas)/len(min_areas))
plt.hist(min_areas, bins=20, edgecolor='black')
plt.title("Distribution of min_area")
plt.xlabel("min_area")
plt.ylabel("count")
plt.show()


import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import distinctipy
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import GATConv
url_living_to_all = r"C:\Users\jensonyu\Documents\ENGR project\Floor_Plan_Generation_using_GNNs-with-boundary\Creating_Dataset\graphs\Graphs_living_to_all_normalized.pkl"
url_boundary = r"C:\Users\jensonyu\Documents\ENGR project\Floor_Plan_Generation_using_GNNs-with-boundary\Creating_Dataset\graphs\boundaries.pkl"


geoms_columns = ['inner', 'living', 'master', 'kitchen', 'bathroom', 'dining', 'child', 'study',
                   'second_room', 'guest', 'balcony', 'storage', 'wall-in',
                    'outer_wall', 'front', 'inner_wall', 'interior',
                   'front_door', 'outer_wall', 'entrance']
N = len(geoms_columns)
colors = (np.array(distinctipy.get_colors(N)) * 255).astype(np.uint8)
room_color = {room_name: colors[i] for i, room_name in enumerate(geoms_columns)}

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
def get_max_min_x_y(graphs):
    max_x = 0
    max_y = 0
    min_x = float('inf')
    min_y = float('inf')
    for G in tqdm(graphs, desc="Getting maximum x, y", total=len(graphs)):
        max_x_in_graph = G.x.T[1].max().item()
        max_y_in_graph = G.x.T[2].max().item()
        min_x_in_graph = G.x.T[1].min().item()
        min_y_in_graph = G.x.T[2].min().item()
        if max_x_in_graph > max_x:
            max_x = max_x_in_graph
        if max_y_in_graph > max_y:
            max_y = max_y_in_graph
        if min_x_in_graph < min_x:
            min_x = min_x_in_graph
        if min_y_in_graph < min_y:
            min_y = min_y_in_graph
    values = {'max_x': max_x, 'max_y': max_y, 'min_x': min_x, 'min_y': min_y}
    return values
def get_all_x_y(graphs):
    """Get all values of x and y from all graphs
        Input: list of graphs
        Output: x and y as pandas series
    """
    x = []
    y = []
    for i, G in tqdm(enumerate(graphs), desc="getting all Xs, Ys", total=len(graphs)):
        for i in range(len(G.x)):
            x.append(G.x[i][1].item())
            y.append(G.x[i][2].item())
    x = pd.Series(x)
    y = pd.Series(y)
    return x, y
def boxplot_centrValues(x, y):
    fig, ax = plt.subplots()
    ax.boxplot([x, y])
    ax.set_xticklabels(['x', 'y'])
    ax.set_xlabel('Data')
    ax.set_ylabel('Value')
    ax.set_title('Boxplot of x and y in all graphs')
    plt.show()
def plot_histograms(x, y):
    x.hist(density=True, bins=100, alpha=0.6, label='x');
    y.hist(density=True, bins=100, alpha=0.3, label='y');
    plt.legend();
    plt.title('Distribution of x and y');

with open(url_living_to_all, 'rb') as f:
    Graphs = pickle.load(f)
G = Graphs[1911]
print(G)

with open(url_boundary, 'rb') as f:
    boundaries = pickle.load(f)
b = boundaries[1911]
print(b)

draw_graph_boundary(b)
draw_graph_nodes(G)

def convert_networkx_Graphs_to_pyTorchGraphs(G):
    """Converting networkx graphs to pytorchGeo graphs with min_area"""
    features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y', 'min_area']
    G_new = from_networkx(G, group_node_attrs=features, group_edge_attrs=['distance'])
    return G_new
Graphs_pyTorch = list(map(convert_networkx_Graphs_to_pyTorchGraphs, Graphs))
Graphs_pyTorch[0]

def convert_networkx_Boundaries_to_pyTorchGraphs(b):
    """Converting networkx boundary graphs to PyTorchGeo graphs
    """
    b_new = from_networkx(b, group_node_attrs=['type', 'centroid'], group_edge_attrs=['distance'])
    return b_new
Boundaries_pyTorch = list(map(convert_networkx_Boundaries_to_pyTorchGraphs, boundaries))
Boundaries_pyTorch[0]

G_x, G_y = get_all_x_y(Graphs_pyTorch)
G_x.max(), G_y.max(), G_x.min(), G_y.min()

boxplot_centrValues(G_x, G_y)

plot_histograms(G_x, G_y)

print("And we saw the box plots so there is no outliers, and the distribution is normal")
G_x_mean = G_x.mean()
G_y_mean = G_y.mean()
G_x_std  = G_x.std()
G_y_std  = G_y.std()
G_min_area = [G.x[:, 3].tolist() for G in Graphs_pyTorch]
G_min_area = pd.Series([item for sublist in G_min_area for item in sublist])
min_area_mean = G_min_area.mean()
min_area_std  = G_min_area.std()
print("We will use z-score normalization")

print(f'Befor: G_1 embedings are: {Graphs_pyTorch[1].x}')
for G in tqdm(Graphs_pyTorch, total=len(Graphs_pyTorch)):
    for j ,value in enumerate(G.x):
        type_ = int(value[0].item())
        if type_ in [1, 4, 5, 6, 7, 8]:
            G.x[j][0] = 1
        elif type_ == 9:
            G.x[j][0] = 4
        elif type_ == 10:
            G.x[j][0] = 5
        elif type_ == 11:
            G.x[j][0] = 6
print(f'After: G_1 embedings are: {Graphs_pyTorch[1].x}')

for G in tqdm(Graphs_pyTorch, total=len(Graphs_pyTorch)):
    G.x[:, 1] = (G.x[:, 1] - G_x_mean) / G_x_std
    G.x[:, 2] = (G.x[:, 2] - G_y_mean) / G_y_std
    G.x[:, 3] = (G.x[:, 3] - min_area_mean) / min_area_std
    first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
    G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)


return_to_real = Graphs_pyTorch[1].x[:, [-3, -2]] * torch.tensor([G_x_std, G_y_std]) + torch.Tensor([G_x_mean, G_y_mean])
print(f"Now, we could return back to real values: \n{return_to_real}")

B_x, B_y = get_all_x_y(Boundaries_pyTorch)
B_x.max(), B_y.max(), B_x.min(), B_y.min()

boxplot_centrValues(B_x, B_y)

plot_histograms(B_x, B_y)

print("And we saw the box plots so there is no outliers, and the distribution is normal")
B_x_mean = B_x.mean()
B_y_mean = B_y.mean()
B_x_std  = B_x.std()
B_y_std  = B_y.std()
print("We will use z-score normalization")

for b in tqdm(Boundaries_pyTorch, total=len(Boundaries_pyTorch)):
    b.x[:, 1:] = (b.x[:, 1:] - torch.tensor([B_x_mean, B_y_mean])) / torch.tensor([B_x_std, B_y_std])

return_to_real = Boundaries_pyTorch[1].x[:, [-2, -1]] * torch.tensor([B_x_std, B_y_std]) + torch.Tensor([B_x_mean, B_y_mean])
print(f"Now, we could return back to real values: \n{return_to_real}")

class Planify_Dataset(Dataset):
    def __init__(self, Graphs, Boundaries):
        self.Graphs = Graphs
        self.Boundaries = Boundaries
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __len__(self):
        return len(self.Graphs)
    def __getitem__(self, index):
        G = self.Graphs[index].clone().to(self.device)
        B = self.Boundaries[index].clone().to(self.device)
        B.x = B.x.to(G.x.dtype)
        B.edge_index = B.edge_index.to(G.edge_index.dtype)
        B.edge_attr = B.edge_attr.to(G.edge_attr.dtype)
        graphs = {
            'G': G,
            'B': B
        }
        return graphs

edge = int(len(Graphs_pyTorch) * 0.8)

batch_size = 32
train_dataset = Planify_Dataset(Graphs_pyTorch[:edge], Boundaries_pyTorch[:edge])
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = Planify_Dataset(Graphs_pyTorch[edge:-10], Boundaries_pyTorch[edge:-10])
val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Planify_Dataset(Graphs_pyTorch[-10:], Boundaries_pyTorch[-10:])
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print(f"Train dataset: {len(train_dataset)}, Val dataset: {len(val_dataset)}, Test dataset: {len(test_dataset)}")

import os
checkpoint_dir = "./checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
def save_checkpoint(model, optimizer, epoch):
    checkpoint_path = os.path.join(checkpoint_dir, f'Best_model_v5.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, checkpoint_path)
    print('Model saved :)')

class GATNet(torch.nn.Module):
    def __init__(self, num_graph_node_features, num_boundary_node_features):
        super(GATNet, self).__init__()
        self.graph_conv1 = GATConv(num_graph_node_features, 32, heads=4)
        input_of_conv2   = num_graph_node_features + 32*4
        self.graph_conv2 = GATConv(input_of_conv2, 32, heads=8)
        input_of_conv3   = num_graph_node_features + 32*8
        self.graph_conv3 = GATConv(input_of_conv3, 64, heads=8)
        input_of_conv4   = num_graph_node_features + 64*8
        self.graph_conv4 = GATConv(input_of_conv4, 128, heads=8)
        shape_of_graphs_befor_concatination = num_graph_node_features + 128*8
        self.boundary_conv1 = GATConv(num_boundary_node_features, 32, heads=4)
        input_of_boundary_conv2 = 32*4 + num_boundary_node_features
        self.boundary_conv2 = GATConv(input_of_boundary_conv2, 32, heads=8)
        shape_of_boundary_befor_concatination = num_boundary_node_features + 32 * 8
        inputs_concatination = shape_of_graphs_befor_concatination + shape_of_boundary_befor_concatination
        self.Concatination1  = GATConv(inputs_concatination, 128, heads=8)
        self.width_layer1  = nn.Linear(128*8, 128)
        self.height_layer1 = nn.Linear(128*8, 128)
        self.width_output  = nn.Linear(128, 1)
        self.height_output = nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, graph, boundary):
        x_graph, g_edge_index, g_edge_attr, g_batch = graph.x, graph.edge_index, graph.edge_attr, graph.batch
        x_boundary, b_edge_indexy, b_edge_attr, b_batch = boundary.x, boundary.edge_index, boundary.edge_attr, boundary.batch
        NUM_OF_NODES = x_graph.shape[0]
        if g_batch == None:
            g_batch = torch.zeros(x_graph.shape[0], dtype=torch.long)
        if b_batch == None:
            b_batch = torch.zeros(x_boundary.shape[0], dtype=torch.long)
        x_graph_res = x_graph
        x_boundary_res = x_boundary
        x_graph = F.leaky_relu(self.graph_conv1(x_graph, g_edge_index, g_edge_attr))
        x_graph = self.dropout(x_graph) # Concatinate with step connection from real values.
        x_graph = torch.cat([x_graph, x_graph_res], dim=1)
        x_graph = F.leaky_relu(self.graph_conv2(x_graph, g_edge_index, g_edge_attr))
        x_graph = self.dropout(x_graph)
        x_graph = torch.cat([x_graph, x_graph_res], dim=1)
        x_graph = F.leaky_relu(self.graph_conv3(x_graph, g_edge_index))
        x_graph = self.dropout(x_graph) 
        x_graph = torch.cat([x_graph, x_graph_res], dim=1)
        x_graph = F.leaky_relu(self.graph_conv4(x_graph, g_edge_index))
        x_graph = self.dropout(x_graph) 
        x_graph = torch.cat([x_graph, x_graph_res], dim=1)
        x_boundary = F.leaky_relu(self.boundary_conv1(x_boundary, b_edge_indexy, b_edge_attr))
        x_boundary = self.dropout(x_boundary)
        x_boundary = torch.cat([x_boundary, x_boundary_res], dim=1)
        x_boundary = F.leaky_relu(self.boundary_conv2(x_boundary, b_edge_indexy, b_edge_attr))
        x_boundary = self.dropout(x_boundary)
        x_boundary = torch.cat([x_boundary, x_boundary_res], dim=1)
        x_boundary_pooled = F.max_pool1d(x_boundary.transpose(0, 1), kernel_size=x_boundary.shape[0]).view(1, -1)
        x = torch.cat([x_graph, x_boundary_pooled.repeat(NUM_OF_NODES, 1)], dim=1)
        x = F.leaky_relu(self.Concatination1(x, g_edge_index))
        x = self.dropout(x)
        width = F.leaky_relu(self.width_layer1(x))
        width = self.dropout(width)
        width = self.width_output(width)
        height = F.leaky_relu(self.height_layer1(x))
        height = self.dropout(height)
        height = self.height_output(height)
        return width.squeeze(), height.squeeze()
num_graph_node_features = Graphs_pyTorch[0].x.shape[1]
num_boundary_node_features = Boundaries_pyTorch[0].x.shape[1]
print("Graph node feature dim:", num_graph_node_features)
print("Boundary node feature dim:", num_boundary_node_features)
model = GATNet(num_graph_node_features, num_boundary_node_features)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
errors = []
acc = []
model

def train(model, optimizer, criterion, train_loader):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        graph, boundary = data['G'], data['B']
        width, height    = model(graph, boundary)
        width_loss = criterion(width, graph.rec_w)
        height_loss = criterion(height, graph.rec_h)
        loss = width_loss + height_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)
def evaluate(model, criterion, val_loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            graph, boundary = data['G'], data['B']
            width, height    = model(graph, boundary)
            width_loss = criterion(width, graph.rec_w)
            height_loss = criterion(height, graph.rec_h)
            loss = width_loss + height_loss
            running_loss += loss.item()
    return running_loss / len(val_loader)

from copy import deepcopy
learning_rate = 0.0005
num_epochs = 250
patience = 50 # Number of epochs to wait if validation loss doesn't improve
best_val_loss = float('inf')
counter = 0
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=3e-5)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.950)
train_losses = []
val_losses = []


for epoch in range(num_epochs):
    train_loss = train(model, optimizer, criterion, train_loader)
    train_losses.append(train_loss)
    print('Validating ...')
    val_loss = evaluate(model, criterion, val_loader)
    val_losses.append(val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = deepcopy(model)
        save_checkpoint(best_model, optimizer, epoch)
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Validation loss did not improve for {patience} epochs. Stopping early.')
            break
        if counter in range(2, 20, 2):
            scheduler.step()
            print(f"Learning rate decreased!, now is {optimizer.state_dict()['param_groups'][0]['lr']}")

plt.plot(train_losses, label=f'Best training    loss: {min(train_losses):.0f}');
plt.plot(val_losses, label=f'Best validation loss: {min(val_losses):.0f}');
plt.legend();

