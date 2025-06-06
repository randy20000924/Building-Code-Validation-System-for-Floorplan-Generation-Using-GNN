from utils import *
from model import *
import json
# from upload import *

def get_info():
    """
        Function to return:
            - Boundary: "POLYGON ((105.44810944559121 78 ....
            - front_door: "POLYGON  (105.44810944559121 78 ....
            - room_centroids: [(81, 105), (55, 151), (134, 105)]
            - bathroom_centroids: [(81, 105), (55, 151), (134, 105)]
            - kitchen_centroids: [(81, 105), (55, 151), (134, 105)]
    """
    boundary_wkt = "POLYGON ((25.599999999999994 65.32413793103447, 200.38620689655173 65.32413793103447, 200.38620689655173 75.91724137931033, 230.4 75.91724137931033, 230.4 190.67586206896553, 67.97241379310344 190.67586206896553, 67.97241379310344 176.55172413793102, 25.599999999999994 176.55172413793102, 25.599999999999994 65.32413793103447))"
    
    front_door_wkt = "POLYGON ((38.436315932155225 179.69850789734912, 63.610586007499926 179.69850789734912, 63.610586007499926 176.55172413793102, 38.436315932155225 176.55172413793102, 38.436315932155225 179.69850789734912))"
    
    # Data of the inner rooms or bathrooms
    room_centroids  = [(201, 163), (193, 106)]
    bathroom_centroids = [(91, 91), (52, 95)]
    kitchen_centroids = [(137, 89)]
    
    # boundary_wkt = input("Enter the boundary as str: ")
    # front_door_wkt = input("Enter the front door as str: ")
    # room_centroids = input("Enter the room centroids as list of tuples: ")
    # bathroom_centroids = input("Enter the bathroom centroids as list of tuples: ")
    # kitchen_centroids = input("Enter the kitchen centroids as list of tuples: ")
    
    return boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids


def preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids):
    Boundary = shapely.wkt.loads(Boundary)
    front_door = shapely.wkt.loads(front_door)

    # Flip the y axis of all polygons and points
    Boundary = scale(Boundary)
    front_door = scale(front_door)
    room_centroids = [scale(x) for x in room_centroids]
    bathroom_centroids = [scale(x) for x in bathroom_centroids]
    kitchen_centroids = [scale(x) for x in kitchen_centroids]

    # Convert to (x, y) tuple
    room_centroids = [x.coords[0] for x in room_centroids]
    bathroom_centroids = [x.coords[0] for x in bathroom_centroids]
    kitchen_centroids = [x.coords[0] for x in kitchen_centroids]
    living_centroid = [(Boundary.centroid.x, Boundary.centroid.y)]

    user_constraints = {
        'living': living_centroid,
        'room': room_centroids,
        'bathroom': bathroom_centroids,
        'kitchen': kitchen_centroids
    }

    # NetworkX graphs
    B_n = Handling_dubplicated_nodes(Boundary, front_door)
    G_n = centroids_to_graph(user_constraints, living_to_all=True)

    # Convert to PyTorch Geometric Data
    B = from_networkx(B_n, group_node_attrs=['type', 'centroid'], group_edge_attrs=['distance'])

    # Add min_area into feature list
    features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y', 'min_area']
    G = from_networkx(G_n, group_edge_attrs=['distance'], group_node_attrs=features)

    # === Normalize x/y only (not min_area) ===
    G_x_mean = G.x[:, 1].mean().item()
    G_y_mean = G.x[:, 2].mean().item()
    G_x_std = G.x[:, 1].std().item()
    G_y_std = G.x[:, 2].std().item()

    G.x[:, 1] = (G.x[:, 1] - G_x_mean) / G_x_std  # x
    G.x[:, 2] = (G.x[:, 2] - G_y_mean) / G_y_std  # y

    # One-hot encode roomType_embd, then concat x, y, min_area
    first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
    G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)

    with open("C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/GAT-Net_model/min_area_stats.json", "r") as f:
        stats = json.load(f)
    G.x[:, -1] = (G.x[:, -1] - stats["mean"]) / stats["std"]

    # === Normalize boundary graph ===
    B_x_mean = B.x[:, 1].mean().item()
    B_y_mean = B.x[:, 2].mean().item()
    B_x_std = B.x[:, 1].std().item()
    B_y_std = B.x[:, 2].std().item()

    B.x[:, 1] = (B.x[:, 1] - B_x_mean) / B_x_std
    B.x[:, 2] = (B.x[:, 2] - B_y_mean) / B_y_std

    # === Tensor type conversion ===
    G.x = G.x.to(torch.float32)
    G.edge_attr = G.edge_attr.to(torch.float32)
    G.edge_index = G.edge_index.to(torch.int64)

    B.x = B.x.to(G.x.dtype)
    B.edge_index = B.edge_index.to(G.edge_index.dtype)
    B.edge_attr = B.edge_attr.to(G.edge_attr.dtype)

    # === Prepare unnormalized version for visualization ===
    B_not_normalized = B.clone()
    G_not_normalized = G.clone()

    G_not_normalized.x[:, -3] = G_not_normalized.x[:, -3] * G_x_std + G_x_mean  # x
    G_not_normalized.x[:, -2] = G_not_normalized.x[:, -2] * G_y_std + G_y_mean  # y

    B_not_normalized.x[:, -2] = B_not_normalized.x[:, -2] * B_x_std + B_x_mean
    B_not_normalized.x[:, -1] = B_not_normalized.x[:, -1] * B_y_std + B_y_mean

    return G, B, G_not_normalized, B_not_normalized, Boundary, front_door, B_n, G_n

    
def Run(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids, output_path):
    # Get the data
    # Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids = get_info()
    
    # ========================================================================
    # Preprocessing
    G, B, G_not_normalized, B_not_normalized, Boundary_as_polygon, front_door_as_polygon, B_n, G_n = preProcessing_toGraphs(Boundary, front_door, room_centroids, bathroom_centroids, kitchen_centroids)
    the_door = Point(B_not_normalized.x[-1][1:].detach().cpu().numpy()).buffer(3)
    
    # geeing the corresponding graph for the inputs of the user
    
    #=========================================================================
    # Model
    # model_path = r"D:\Grad\Best models\v2\Best_model_V2.pt"
    # model_path = r"C:/Users/jensonyu\Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/GAT-Net_model/checkpoints/GATNet_no_share_weights.pt"
    model_path = r"C:/Users/jensonyu\Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/GAT-Net_model/checkpoints/Best_model_v4.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path, device, 10, 3)
    
    #=========================================================================
    # Inference
    prediction    = model(G.to(device), B.to(device))
    w_predicted   = prediction[0].detach().cpu().numpy()
    h_predicted   = prediction[1].detach().cpu().numpy()
    prediction    = np.concatenate([w_predicted.reshape(-1, 1), h_predicted.reshape(-1, 1)], axis=1)
    
    #=========================================================================
    # Rescaling back to the original values
    # G.x[:, -2] = G.x[:, -2] * G_x_std + G_x_mean
    #=========================================================================
    # Visualization
    output = FloorPlan_multipolygon(G_not_normalized, prediction)
    polygons = output.get_multipoly(Boundary_as_polygon, the_door)
    fig, ax = plt.subplots(figsize=(4, 4))
    polygons.plot(ax=ax, cmap='twilight', alpha=0.8, linewidth=0.8, edgecolor='black')
    # polygons.plot(cmap='twilight', figsize=(4, 4), alpha=0.8, linewidth=0.8, edgecolor='black')
    
    #=========================================================================
    # Saving the output & Updating to the firebase
    # unique_name = str(uuid.uuid4())
    # if not os.path.exists("./Outputs"):
    #     os.mkdir("Outputs/" + unique_name)
    # plt.savefig("Outputs/" + '/Output.png')
    # image_url = upload_to_firebase(unique_name)
    # print(image_url)
    # print("Done")

    # === 加比例尺 ===
    # 每 unit 是 0.5 公尺，畫一段代表 10 公尺的比例尺（10m = 20 units）
    scale_length_in_unit = 20
    scale_length_in_m = scale_length_in_unit * 0.5

    # 選擇左下角當作比例尺位置
    minx, miny, maxx, maxy = polygons.total_bounds
    scale_x_start = minx + 5
    scale_y_pos = miny + 5

    # 畫線段（比例尺）
    ax.plot([scale_x_start, scale_x_start + scale_length_in_unit], [scale_y_pos, scale_y_pos], 
            color='black', linewidth=2)

    # 加文字
    ax.text(scale_x_start, scale_y_pos + 2, f'{scale_length_in_m} m', fontsize=8)
    
    os.makedirs("C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/Outputs", exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    # Inference
    prediction = model(G.to(device), B.to(device))
    w_predicted = prediction[0].detach().cpu().numpy()
    h_predicted = prediction[1].detach().cpu().numpy()
    prediction = np.concatenate([w_predicted.reshape(-1, 1), h_predicted.reshape(-1, 1)], axis=1)

    return output_path, B_n, G_n

def get_example(example_name):
    if example_name == 'EX 1':
        boundary_wkt = "POLYGON ((25.599999999999994 65.32413793103447, 200.38620689655173 65.32413793103447, 200.38620689655173 75.91724137931033, 230.4 75.91724137931033, 230.4 190.67586206896553, 67.97241379310344 190.67586206896553, 67.97241379310344 176.55172413793102, 25.599999999999994 176.55172413793102, 25.599999999999994 65.32413793103447))"
    
        front_door_wkt = "POLYGON ((38.436315932155225 179.69850789734912, 63.610586007499926 179.69850789734912, 63.610586007499926 176.55172413793102, 38.436315932155225 176.55172413793102, 38.436315932155225 179.69850789734912))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(201, 163), (193, 106)]
        bathroom_centroids = [(91, 91), (52, 95)]
        kitchen_centroids = [(137, 89)]
    
    elif example_name == 'EX 2':
        boundary_wkt = "POLYGON ((29.043431083399053 47.85822973601723, 179.28640942886287 47.85822973601723, 179.28640942886287 75.73837004754658, 230.4 75.73837004754658, 230.4 208.94348486929798, 58.47246807890228 208.94348486929798, 58.47246807890228 162.4765843500824, 29.043431083399053 162.4765843500824, 29.043431083399053 47.85822973601723))"
        
        front_door_wkt = "POLYGON ((32.48686216679811 47.05651513070204, 25.599999999999994 47.05651513070204, 25.599999999999994 74.60396379789447, 29.043431083399053 74.60396379789447, 29.043431083399053 47.85822973601723, 32.48686216679811 47.85822973601723, 32.48686216679811 47.05651513070204))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(203, 101), (191, 171)]
        bathroom_centroids = [(83, 182), (152, 66)]
        kitchen_centroids = [(52, 131)]
        
    elif example_name == 'EX 3':
        
        boundary_wkt = "POLYGON ((44.92075471698112 53.615094339622644, 230.4 53.615094339622644, 230.4 78.73207547169811, 209.14716981132074 78.73207547169811, 209.14716981132074 202.38490566037737, 25.599999999999994 202.38490566037737, 25.599999999999994 71.00377358490566, 44.92075471698112 71.00377358490566, 44.92075471698112 53.615094339622644))"
        
        front_door_wkt = "POLYGON ((212.46404317939437 194.3995689439927, 212.46404317939437 167.86458199940353, 209.14716981132074 167.86458199940353, 209.14716981132074 194.3995689439927, 212.46404317939437 194.3995689439927))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(184, 77), (67, 94)]
        bathroom_centroids = [(185, 134), (126, 74)]
        kitchen_centroids = [(52, 176)]

    elif example_name == 'EX 4':
        boundary_wkt = "POLYGON ((58.18181818181817 69.85672370603027, 230.4 69.85672370603027, 230.4 105.54157219087877, 211.7818181818182 105.54157219087877, 211.7818181818182 200.183996433303, 25.599999999999994 200.183996433303, 25.599999999999994 58.99611764542421, 58.18181818181817 58.99611764542421, 58.18181818181817 69.85672370603027))"
        
        front_door_wkt = "POLYGON ((56.16288055733307 55.816003566697006, 30.721967927515397 55.816003566697006, 30.721967927515397 58.99611764542421, 56.16288055733307 58.99611764542421, 56.16288055733307 55.816003566697006))"
        
        # Data of the inner rooms or bathrooms
        room_centroids  = [(198, 87), (174, 166)]
        bathroom_centroids = [(51, 169), (148, 91)]
        kitchen_centroids = [(44, 105)]
        
        
    return boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids


if __name__ == '__main__':

    # ex = get_example('EX 3')

    # # for testing
    # boundary_wkt = ex[0]
    # front_door_wkt = ex[1]
    # room_centroids  = ex[2]
    # bathroom_centroids = ex[3]
    # kitchen_centroids = ex[4]
    # output_path = "C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/Outputs/model_output_test.png"

    # Run(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids, output_path)

    for i in range(1, 5):
        ex = get_example('EX '+str(i))
        boundary_wkt = ex[0]
        front_door_wkt = ex[1]
        room_centroids  = ex[2]
        bathroom_centroids = ex[3]
        kitchen_centroids = ex[4]
        output_path = "C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/Outputs/model_output_test"+ str(i) +".png"
        Run(boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids, output_path)

    # Run()