# Import necessary libraries
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

# Suppress OpenMP warning
os.environ['OMP_NUM_THREADS'] = '1'

# Step 1: Load and preprocess the immunotherapy therapy regimens dataset
def load_therapy_regimens(file_path):
    """
    Load and preprocess the immunotherapy therapy regimens dataset.
    """
    # Load dataset
    therapy_data = pd.read_excel(file_path)

    # Define column mappings
    required_columns = {
        'Therapy_Regimen': 'Therapy Regimen',
        'Target_Proteins': 'Target Proteins',
        'Approval_Status': 'Approved/In Clinical Trials',
        'Clinical_Trial_Phase': 'Phase',
        'Disease': 'Disease'
    }

    # Ensure required columns exist
    for new_col, old_col in required_columns.items():
        if old_col not in therapy_data.columns:
            raise ValueError(f"Column '{old_col}' not found in dataset.")

    # Rename columns for consistency
    therapy_data = therapy_data.rename(columns={v: k for k, v in required_columns.items() if v})

    # Drop missing values
    therapy_data = therapy_data.dropna(subset=['Therapy_Regimen', 'Target_Proteins', 'Approval_Status', 'Clinical_Trial_Phase', 'Disease'])

    print(f"Loaded {len(therapy_data['Therapy_Regimen'].unique())} therapy regimens.")
    return therapy_data

# Step 2: Load and preprocess scRNA-seq data
def load_scRNA_data(file_path):
    """
    Load and preprocess scRNA-seq data.
    """
    adata = sc.read(file_path)  # Load scRNA-seq data
    sc.pp.filter_cells(adata, min_genes=200)  # Filter cells with fewer than 200 genes
    sc.pp.filter_genes(adata, min_cells=3)  # Filter genes expressed in fewer than 3 cells
    sc.pp.normalize_total(adata, target_sum=1e4)  # Normalize data
    sc.pp.log1p(adata)  # Log-transform data
    return adata

# Step 3: Model the Tumor Microenvironment (TME) using Graph Neural Networks (GNNs)
class GNNModel(nn.Module):
    """
    Graph Neural Network to model cell-cell interactions in the TME.
    """
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
    """
    Create a graph representation of the Tumor Microenvironment (TME).
    """
    # Convert gene expression to tensor
    X = torch.tensor(adata.X, dtype=torch.float)

    # Create a fully connected graph (simplified approach)
    num_samples = X.shape[0]
    edge_index = torch.tensor([[i, j] for i in range(num_samples) for j in range(num_samples) if i != j], dtype=torch.long).t()

    return Data(x=X, edge_index=edge_index)

def train_gnn(data, model, epochs=50):
    """
    Train the GNN model.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.x)  # Reconstruct input features
        loss.backward()
        optimizer.step()

        # Print loss every 10 epochs and at the final epoch
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

# Step 4: Custom Gym Environment for RL-based Immunotherapy Optimization
class TMEEnv(gym.Env):
    """
    Custom environment for reinforcement learning to optimize immunotherapy.
    """
    def __init__(self, adata, therapy_data, gnn_model):
        super(TMEEnv, self).__init__()

        self.adata = adata
        self.therapy_data = therapy_data
        self.gnn_model = gnn_model
        self.n_actions = len(therapy_data['Therapy_Regimen'].unique())

        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_actions)
        self.state_size = gnn_model.output_dim  # Use GNN output dimension as state size
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # Initialize state
        self.state = self._get_initial_state()
        self.current_step = 0

    def _get_initial_state(self):
        """
        Use GNN to generate initial state from TME data.
        """
        data = create_tme_graph(self.adata)
        with torch.no_grad():
            state = self.gnn_model(data.x, data.edge_index).mean(dim=0).numpy()  # Aggregate GNN outputs
        return state

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_step = 0
        self.state = self._get_initial_state()
        return self.state

    def step(self, action):
        """
        Step function that evaluates an action (selected therapy regimen) based on multiple biological factors.
        """
        therapy_regimen = self.therapy_data['Therapy_Regimen'].unique()[action]
        therapy_info = self.therapy_data[self.therapy_data['Therapy_Regimen'] == therapy_regimen]

        # Fetch relevant factors from the dataset
        approval_status = therapy_info['Approval_Status'].mode()[0]
        phase = therapy_info['Clinical_Trial_Phase'].mode()[0]

        # Use GNN to predict biological factors
        data = create_tme_graph(self.adata)
        with torch.no_grad():
            gnn_output = self.gnn_model(data.x, data.edge_index).mean(dim=0).numpy()  # Aggregate GNN outputs

        # Extract GNN predictions
        expression_level = gnn_output[0]  # Assume first dimension represents expression level
        immune_cell_presence = gnn_output[1]  # Assume second dimension represents immune cell presence
        immune_escape_risk = gnn_output[2]  # Assume third dimension represents immune escape risk

        # **Initialize Reward**
        reward = 0

        # **Clinical Trial Phase Rewards**
        phase_rewards = {"Phase I": 2, "Phase II": 5, "Phase III": 8, "Approved": 10}
        reward += phase_rewards.get(phase, 0)

        # **Disease Specificity Bonus**
        disease = self.current_disease  # Assuming you track the disease in the environment
        if therapy_regimen in known_effective_therapies[disease]:
            reward += 3  # Bonus if the therapy is known to work for this disease

        # **Balance Risk vs. Reward**
        if phase in ["Phase I", "Phase II"]:
            reward -= 1  # Small penalty for early-stage therapies

        # **Additional Biological Factors**
        reward += 0.5 * expression_level  # Favor therapies targeting highly expressed proteins
        reward += 0.3 * immune_cell_presence  # Reward therapies that increase immune cell presence
        reward -= 0.5 * immune_escape_risk  # Penalize therapies that increase immune escape risk

        # Update state with GNN predictions
        self.state = gnn_output
        self.current_step += 1
        done = self.current_step >= 10

        return self.state, reward, done, {}

# Step 5: Optimize Immunotherapy using RL
def optimize_immunotherapy(adata, therapy_data, gnn_model):
    """
    Train RL model to optimize immunotherapy selection.
    """
    env = DummyVecEnv([lambda: TMEEnv(adata, therapy_data, gnn_model)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=1000)
    return model

# Step 6: Explain Recommendations
def explain_recommendations(adata, model, therapy_data):
    """
    Use SHAP to explain the AI's recommendations based on therapy-specific features.
    """
    # Encode categorical columns
    therapy_data['Therapy_Regimen'] = therapy_data['Therapy_Regimen'].astype('category').cat.codes
    therapy_data['Approval_Status'] = therapy_data['Approval_Status'].astype('category').cat.codes

    # Prepare data for SHAP
    data_for_shap = therapy_data[['Therapy_Regimen', 'Approval_Status']].copy()

    # Create a wrapper function for the RL model
    def predict_fn(data):
        return rl_predict_wrapper(model, data)

    # Initialize SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, data=data_for_shap)

    # Compute SHAP values
    shap_values = explainer.shap_values(data_for_shap)

    print("SHAP values computed for model interpretability.")

    # Visualize SHAP values
    shap.summary_plot(shap_values, data_for_shap)

# Main function
def main():
    # Step 1: Load immunotherapy therapy regimens dataset
    therapy_file_path = 'Cancer Immunotherapy Therapy Regimens.xlsx'
    therapy_data = load_therapy_regimens(therapy_file_path)

    # Step 2: Load and preprocess scRNA-seq data
    scRNA_file_path = 'CellXgene Dataset.h5ad'  # Replace with your scRNA-seq data file
    adata = load_scRNA_data(scRNA_file_path)

    # Step 3: Model the Tumor Microenvironment (TME) using GNNs
    data = create_tme_graph(adata)
    gnn_model = GNNModel(input_dim=data.num_features, hidden_dim=8, output_dim=3)  # Output 3 features: expression, immune cell presence, immune escape risk
    train_gnn(data, gnn_model, epochs=50)

    # Step 4: Optimize Immunotherapy using RL
    rl_model = optimize_immunotherapy(adata, therapy_data, gnn_model)

    # Step 5: Explain AI Recommendations
    explain_recommendations(adata, rl_model, therapy_data)

if __name__ == "__main__":
    main()