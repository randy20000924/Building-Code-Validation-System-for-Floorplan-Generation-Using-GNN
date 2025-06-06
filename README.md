🏗️ Building Code Validation System for Floorplan Generation Using GNN

This project aims to validate architectural floorplans against building codes using a Graph Neural Network (GNN)-based model. By encoding spatial and semantic relationships between rooms, we enable intelligent verification of constraints such as adjacency, accessibility, and required feature presence.

📌 Motivation
Automated floorplan generation is a growing area in architectural and generative design tools. However, ensuring regulatory compliance remains a significant challenge. This project introduces a machine learning-based validation system that learns to identify violations from labeled samples, providing a scalable alternative to rule-based logic.

🧠 Core Idea
Input: Floorplan as a graph

Nodes = rooms (with type, size, etc.)

Edges = spatial or functional adjacency

Output: Binary classification for each building code (pass/fail)

We use GNNs to learn high-level spatial rules and predict whether the floorplan adheres to given regulations.

🏗️ Architecture
Graph Construction:
Convert floorplan (JSON/CAD/XML) into a node-edge graph structure

GNN Model:
A custom GCN/GAT model built with PyTorch Geometric

Validation Codes:
Encoded constraints include:

Bedroom adjacency to at least one bathroom

Kitchen access from common areas

Minimum size for rooms by type

Fire escape distances (if available)

📊 Dataset
Real and synthetic floorplans

Labeled compliance information based on predefined code rules

Format: JSON → Graph

Dataset preprocessing scripts are in /data folder
