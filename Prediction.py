import os
import torch
import glob
import numpy as np
from tqdm import tqdm
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GlobalAttention
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import LayerNorm
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_max_pool, global_mean_pool

params = {
    'LABEL_EXCEL': "/home/knu_bs/project/Dataset_KCC/DATA/DATA.xlsx",
    'LABEL_COLUMN': "pKa",
    'GRAPH_DIR': "/home/knu_bs/project/Dataset_KCC/DATA/GR 7.14",
    'DEVICE': torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    'IN_DIM': 1280,
    'HIDDEN_DIM': 256,
    'OUT_DIM': 512,
    'BATCH_SIZE': 32,
    'EPOCHS': 100,
    'LEARNING_RATE': 0.0005,
    'WEIGHT_DECAY': 1e-3,
    'DROPOUT': 0.2,
    'NUM_LAYERS': 7
}

class GraphTripletDataset(Dataset):
    def __init__(self, graph_dir):
        self.label_df = pd.read_excel(params['LABEL_EXCEL'])
        self.label_map = dict(zip(self.label_df['PDB ID'], self.label_df[params['LABEL_COLUMN']]))
        self.pdb_dict = self._group_by_pdb(graph_dir)
        self.pdb_ids = list(self.pdb_dict.keys())

    def _group_by_pdb(self, graph_dir):
        pt_files = glob.glob(os.path.join(graph_dir, '*.pt'))
        pdb_dict = {}
        for path in pt_files:
            fname = os.path.basename(path)
            pdb_id = fname.split('_')[0]
            pdb_dict.setdefault(pdb_id, {'intra': [], 'inter': []})
            if 'INTER' in fname.upper():
                pdb_dict[pdb_id]['inter'].append(path)
            else:
                pdb_dict[pdb_id]['intra'].append(path)
        return {k: v for k, v in pdb_dict.items() if len(v['intra']) == 2 and len(v['inter']) == 1}

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        pdb_id = self.pdb_ids[idx]
        paths_intra = self.pdb_dict[pdb_id]['intra']
        paths_inter = self.pdb_dict[pdb_id]['inter']
        reps = []
        for pt_path in paths_intra + paths_inter:
            pt = torch.load(pt_path)
            data = Data(x=pt['node_features'], pos=pt['coords'], edge_index=pt['edge_index'])
            reps.append(data)
        label = torch.tensor([self.label_map.get(pdb_id, 0.0)], dtype=torch.float)
        return reps, label, pdb_id

class EGNN_Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, 1),
            nn.ReLU(),
            nn.Tanh()
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(dim + dim, dim),
            nn.ReLU()
        )
        self.norm = LayerNorm(dim)
        self.coord_res_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, h, x, edge_index):
        row, col = edge_index
        rel_pos = x[row] - x[col]
        dist2 = torch.sum(rel_pos ** 2, dim=-1, keepdim=True)

        edge_feat = torch.cat([h[row], h[col], dist2], dim=-1)
        m_ij = self.edge_mlp(edge_feat)

        scale = self.coord_mlp(m_ij)
        coord_update = rel_pos * scale

        delta = torch.zeros_like(x)
        delta.index_add_(0, row, coord_update)
        x = x + self.coord_res_weight * delta

        agg = torch.zeros_like(h)
        agg.index_add_(0, row, m_ij)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1))
        h = self.norm(h + h_new)  # Residual + LayerNorm

        return h, x

class EGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([EGNN_Layer(hidden_dim) for _ in range(num_layers)])

        self.output = nn.Linear(hidden_dim, out_dim)  

    def forward(self, x, pos, edge_index, batch):
        h = self.embedding(x)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
            
        h_pool = global_max_pool(h, batch)

        return self.output(h_pool), pos


class InteractionPredictor(nn.Module):
    def __init__(self, gnn_out_dim):
        super().__init__()
        d = params['DROPOUT']
        self.mlp = nn.Sequential(
            nn.Linear(gnn_out_dim * 3, 512), nn.ReLU(), nn.Dropout(d),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(d),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(d),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(d),
            nn.Linear(32, 1)
        )

    def forward(self, rep1, rep2, rep3):
        return self.mlp(torch.cat([rep1, rep2, rep3], dim=-1)).squeeze()

def compute_rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2)).item()

def compute_r2(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    return 1 - ss_res / ss_tot

def compute_pearsonr(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return np.corrcoef(preds, labels)[0, 1]

def get_batch(graph):
    return graph.batch if hasattr(graph, 'batch') else torch.zeros(graph.num_nodes, dtype=torch.long).to(graph.x.device)

def get_rep(graph, model):
    return model(graph.x, graph.pos, graph.edge_index, get_batch(graph))[0]

def save_predictions_to_csv(pdb_ids, preds, labels, filename="pKd_predictions1.csv"):
    df = pd.DataFrame({"PDB_ID": pdb_ids, "True_pKd": labels, "Predicted_pKd": preds})
    df.to_csv(filename, index=False)
    print(f"✅ Saved predictions to {filename}")

def plot_prediction_scatter(preds, labels, filename="pKd_scatter_plot2115.png"):
    preds = np.array(preds)
    labels = np.array(labels)
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    mae = np.mean(np.abs(preds - labels))
    r = np.corrcoef(preds, labels)[0, 1] ## 이거야 

    # 📏 8x8인치 정사각형 고해상도 출력
    plt.figure(figsize=(8.0, 4.5))
    plt.scatter(labels, preds, alpha=0.6, s=10)
    plt.plot([min(labels), max(labels)], [min(labels), max(labels)], '--', color='gray')
    plt.xlabel(r"True $\mathrm{p}K_D$")
    plt.ylabel(r"Predicted $\mathrm{p}K_D$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    print(f"📊 Scatter plot saved to {filename}")


def train_model():
    dataset = GraphTripletDataset(params['GRAPH_DIR'])
    train_ids, temp_ids = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    train_loader = DataLoader([dataset[i] for i in train_ids], batch_size=params['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader([dataset[i] for i in val_ids], batch_size=1)
    test_loader = DataLoader([dataset[i] for i in test_ids], batch_size=1)

    egnn = EGNN(params['IN_DIM'], params['HIDDEN_DIM'], params['OUT_DIM'], params['NUM_LAYERS']).to(params['DEVICE'])
    predictor = InteractionPredictor(params['OUT_DIM']).to(params['DEVICE'])
    optimizer = optim.Adam(list(egnn.parameters()) + list(predictor.parameters()),
                           lr=params['LEARNING_RATE'], weight_decay=params['WEIGHT_DECAY'])
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir="runs/graph_interaction")

    best_val_rmse = float('inf')

    # 🧪 테스트 ID 저장
    test_pdb_ids = [dataset.pdb_ids[i] for i in test_ids]
    df = pd.DataFrame({'PDB_ID': test_pdb_ids})
    df.to_csv("test_pdb_ids_seed42.csv", index=False)

    for epoch in range(params['EPOCHS']):
        egnn.train()
        predictor.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            reps, label, _ = batch
            reps = [g.to(params['DEVICE']) for g in reps]
            label = label.to(params['DEVICE'])

            rep1 = get_rep(reps[0], egnn)
            torch.cuda.empty_cache()
            rep2 = get_rep(reps[1], egnn)
            torch.cuda.empty_cache()
            rep3 = get_rep(reps[2], egnn)
            torch.cuda.empty_cache()

            pred = predictor(rep1, rep2, rep3)
            loss = criterion(pred.squeeze(), label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/MSE_Loss', avg_loss, epoch)

        # Validation
        egnn.eval()
        predictor.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                reps, label, _ = batch
                reps = [g.to(params['DEVICE']) for g in reps]
                label = label.to(params['DEVICE'])
                rep1 = get_rep(reps[0], egnn)
                torch.cuda.empty_cache()
                rep2 = get_rep(reps[1], egnn)
                torch.cuda.empty_cache()
                rep3 = get_rep(reps[2], egnn)
                torch.cuda.empty_cache()
                pred = predictor(rep1, rep2, rep3)
                val_preds.append(pred.squeeze())
                val_labels.append(label.squeeze())

        val_preds = torch.stack(val_preds)
        val_labels = torch.stack(val_labels)
        val_rmse = compute_rmse(val_preds, val_labels)
        val_mae = torch.mean(torch.abs(val_preds - val_labels)).item()
        val_mse = torch.mean((val_preds - val_labels)**2).item()
        val_r2 = compute_r2(val_preds, val_labels)

        writer.add_scalar('Validation/MSE_Loss', val_mse, epoch)
        writer.add_scalar('Validation/R2', val_r2, epoch)

        print(f"Epoch {epoch+1}: Train MSE = {avg_loss:.4f}, Validation MSE = {val_mse:.4f}, RMSE = {val_rmse:.4f}, MAE = {val_mae:.4f}, R = {compute_pearsonr(val_preds, val_labels):.4f}, R² = {val_r2:.4f}")

    egnn.eval()
    predictor.eval()
    test_preds, test_labels, test_pdb_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            reps, label, pdb_id = batch
            reps = [g.to(params['DEVICE']) for g in reps]
            label = label.to(params['DEVICE'])
            rep1 = get_rep(reps[0], egnn)
            torch.cuda.empty_cache()
            rep2 = get_rep(reps[1], egnn)
            torch.cuda.empty_cache()
            rep3 = get_rep(reps[2], egnn)
            torch.cuda.empty_cache()
            pred = predictor(rep1, rep2, rep3)
            test_preds.append(pred.squeeze().item())
            test_labels.append(label.squeeze().item())
            test_pdb_ids.append(pdb_id[0])

    test_rmse = compute_rmse(torch.tensor(test_preds), torch.tensor(test_labels))
    test_mae = torch.mean(torch.abs(torch.tensor(test_preds) - torch.tensor(test_labels))).item()
    test_r2 = compute_r2(torch.tensor(test_preds), torch.tensor(test_labels))
    test_r = compute_pearsonr(torch.tensor(test_preds), torch.tensor(test_labels)) 

    print(f"\n📊 Test Set Performance:\nRMSE = {test_rmse:.4f}, MAE = {test_mae:.4f}, R² = {test_r2:.4f}, R = {test_r:.4f}") 
    save_predictions_to_csv(test_pdb_ids, test_preds, test_labels)
    plot_prediction_scatter(test_preds, test_labels)
    writer.close()

if __name__ == "__main__":
    train_model()

