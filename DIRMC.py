#!/usr/bin/env python3
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import shap
import gym
from gym import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Suppress warnings and OpenMP messages
warnings.filterwarnings("ignore")
os.environ['OMP_NUM_THREADS'] = '1'

# --------------------------------------------------
# Step 1: Load and Preprocess the Immunotherapy Targets Dataset
def load_immunotherapy_targets(file_path):
    targets_data = pd.read_excel(file_path)
    required_columns = {
        'Target_Protein': 'Target/Antigen(s)',
        'Approval_Status': 'Approved/In Clinical Trials',
        'Clinical_Trial_Phase': 'Phase',
        'Disease': 'Disease'
    }
    for new_col, old_col in required_columns.items():
        if old_col not in targets_data.columns:
            raise ValueError(f"Column '{old_col}' not found in dataset.")
    targets_data = targets_data.rename(columns={v: k for k, v in required_columns.items()})
    targets_data = targets_data.dropna(subset=['Target_Protein', 'Approval_Status', 'Clinical_Trial_Phase', 'Disease'])
    print(f"Loaded {len(targets_data['Target_Protein'].unique())} immunotherapy targets.")
    return targets_data

# --------------------------------------------------
# Step 2: Define the GNN Model and Helper Functions
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_tme_graph(adata):
    X = torch.tensor(adata.X, dtype=torch.float)
    num_samples = X.shape[0]
    # Create a fully connected graph (excluding self-loops)
    edge_index = torch.tensor(
        [[i, j] for i in range(num_samples) for j in range(num_samples) if i != j],
        dtype=torch.long
    ).t().contiguous()
    return Data(x=X, edge_index=edge_index)

def train_gnn(data, model, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# --------------------------------------------------
# Step 3: Define the Custom Gym Environment for RL
class TMEEnv(gym.Env):
    def __init__(self, adata, targets_data):
        super(TMEEnv, self).__init__()
        self.adata = adata
        self.targets_data = targets_data
        self.n_actions = len(targets_data['Target_Protein'].unique())
        # Define a simple observation space (here, 100-dimensional)
        n_env = 100
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_env,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.state = np.random.rand(n_env)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        self.state = np.random.rand(self.observation_space.shape[0])
        return self.state

    def step(self, action):
        target_protein = self.targets_data['Target_Protein'].unique()[action]
        approval_status = self.targets_data[self.targets_data['Target_Protein'] == target_protein]['Approval_Status'].mode()[0]
        phase = self.targets_data[self.targets_data['Target_Protein'] == target_protein]['Clinical_Trial_Phase'].mode()[0]

        # Reward logic based on approval and clinical phase
        if approval_status == 'Approved':
            reward = 1.0
        elif phase == 'Phase III':
            reward = 0.8
        elif phase == 'Phase II':
            reward = 0.6
        else:
            reward = 0.4

        self.state = self.state + np.random.normal(0, 0.01, size=self.state.shape)
        self.current_step += 1
        done = self.current_step >= 10
        return self.state, reward, done, {}

# --------------------------------------------------
# Step 4: Optimize Immunotherapy using an RL Model (PPO)
def optimize_immunotherapy(adata, targets_data):
    env = DummyVecEnv([lambda: TMEEnv(adata, targets_data)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)
    return model

# --------------------------------------------------
# Step 5: Predict Immune Escape (Placeholder Function)
def predict_immune_escape(adata, targets_data):
    print("Predicting immune escape and adapting therapy...")

# --------------------------------------------------
# Step 6: Explainability via SHAP
def rl_predict_wrapper(model, data):
    # Prepare dummy observations matching the environment's shape
    observations = np.zeros((data.shape[0], 100))
    predictions = []
    for i in range(data.shape[0]):
        action, _ = model.predict(observations[i], deterministic=True)
        predictions.append(action)
    return np.array(predictions)

def explain_recommendations(adata, model, targets_data):
    # Convert categorical data for SHAP analysis
    targets_data['Target_Protein'] = targets_data['Target_Protein'].astype('category').cat.codes
    targets_data['Approval_Status'] = targets_data['Approval_Status'].astype('category').cat.codes
    data_for_shap = targets_data[['Target_Protein', 'Approval_Status']].copy()

    def predict_fn(data):
        return rl_predict_wrapper(model, data)

    explainer = shap.KernelExplainer(predict_fn, data=data_for_shap)
    shap_values = explainer.shap_values(data_for_shap)
    print("SHAP values computed for model interpretability.")
    shap.summary_plot(shap_values, data_for_shap)

# --------------------------------------------------
# Terminal-Based Chatbot Loop (Integrating with Ollama)
def chatbot_loop(rl_model, adata, targets_data):
    print("Welcome to the Immunotherapy Optimization Chatbot (Terminal-based with Ollama)!")
    print("Available commands:")
    print("  recommend - Get an immunotherapy recommendation")
    print("  predict   - Predict immune escape mechanisms")
    print("  explain   - Explain AI recommendations")
    print("  quit      - Exit the chatbot")
    
    while True:
        user_input = input("You: ").strip().lower()
        if user_input == "quit":
            print("Exiting chatbot. Goodbye!")
            break
        elif user_input == "recommend":
            # Create a new environment instance for prediction
            env = TMEEnv(adata, targets_data)
            state = env.reset()
            action, _ = rl_model.predict(state, deterministic=True)
            recommendation = targets_data['Target_Protein'].unique()[action]
            print(f"Recommendation: {recommendation}")
        elif user_input == "predict":
            predict_immune_escape(adata, targets_data)
        elif user_input == "explain":
            explain_recommendations(adata, rl_model, targets_data)
        else:
            print("Unrecognized command. Please use 'recommend', 'predict', 'explain', or 'quit'.")

# --------------------------------------------------
# Main function to run the full pipeline and launch the chatbot
def main():
    # Load the immunotherapy targets dataset
    targets_file_path = './Cancer immunotherapy targets approved and undergoing clinical trials.xlsx'
    targets_data = load_immunotherapy_targets(targets_file_path)
    
    # Create a sample scRNA-seq dataset (using random data for demonstration)
    adata = sc.AnnData(np.random.rand(100, 50))  # 100 samples, 50 genes
    
    # Build the TME graph and train the GNN model
    data = create_tme_graph(adata)
    gnn_model = GNNModel(input_dim=data.num_features, hidden_dim=8, output_dim=data.num_features)
    train_gnn(data, gnn_model, epochs=50)
    
    # Optimize immunotherapy using RL (PPO)
    rl_model = optimize_immunotherapy(adata, targets_data)
    
    # Launch the terminal-based chatbot
    chatbot_loop(rl_model, adata, targets_data)

if __name__ == "__main__":
    main()
