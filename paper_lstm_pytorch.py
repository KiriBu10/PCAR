import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import time
import pickle
from sklearn.preprocessing import LabelEncoder
from framework.utils import compute_mae, compute_accuracy, compute_f1

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Dataset class
class PrefixDataset(Dataset):
    def __init__(self, sequences, activity_labels, time_labels, max_len, num_activities):
        self.sequences = [torch.tensor(seq + [0] * (max_len - len(seq))) for seq in sequences]
        self.activity_labels = activity_labels
        self.time_labels = time_labels
        self.max_len = max_len
        self.num_activities = num_activities

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = F.one_hot(self.sequences[idx], num_classes=self.num_activities).float()
        return seq, self.activity_labels[idx], self.time_labels[idx]

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, num_activities, hidden_dim=100):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(num_activities, hidden_dim, batch_first=True)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.lstm2_1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn2_1 = nn.LayerNorm(hidden_dim)
        self.lstm2_2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.bn2_2 = nn.LayerNorm(hidden_dim)

        self.fc_activity = nn.Linear(hidden_dim, num_activities)
        self.fc_time = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x_bn = self.bn1(x[:, -1, :])
        x1, _ = self.lstm2_1(x)
        x1_bn = self.bn2_1(x1[:, -1, :])
        x2, _ = self.lstm2_2(x)
        x2_bn = self.bn2_2(x2[:, -1, :])

        activity_out = self.fc_activity(x1_bn)
        time_out = self.fc_time(x2_bn).squeeze(1)
        return activity_out, time_out

def run_experiments_pytorch(n_runs, train_df, val_df, test_df, evaluation_df, save_path_prefix="models/lstm_run"):
    results = []

    MAX_LEN = max(len(p) for p in evaluation_df['prefix_int'])
    all_activity_ids = [item for sublist in evaluation_df['prefix_int'] for item in sublist]
    NUM_ACTIVITIES = max(all_activity_ids) + 1

    for run in range(1, n_runs + 1):
        print(f"\n=== Run {run} ===")

        # Prepare datasets
        train_dataset = PrefixDataset(train_df['prefix_int'], train_df['next_activity_int'].values, train_df['remaining_time'].values, MAX_LEN, NUM_ACTIVITIES)
        val_dataset = PrefixDataset(val_df['prefix_int'], val_df['next_activity_int'].values, val_df['remaining_time'].values, MAX_LEN, NUM_ACTIVITIES)
        test_dataset = PrefixDataset(test_df['prefix_int'], test_df['next_activity_int'].values, test_df['remaining_time'].values, MAX_LEN, NUM_ACTIVITIES)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        model = LSTMModel(NUM_ACTIVITIES).to(device)
        criterion_activity = nn.CrossEntropyLoss()
        criterion_time = nn.L1Loss()
        optimizer = torch.optim.NAdam(model.parameters(), lr=0.002)

        early_stop_patience = 15
        best_val_loss = float('inf')
        patience_counter = 0

        start_time = time.time()
        for epoch in range(100):
            model.train()
            for batch in train_loader:
                inputs, act_labels, time_labels = [b.to(device) for b in batch]
                optimizer.zero_grad()
                act_out, time_out = model(inputs)
                loss_act = criterion_activity(act_out, act_labels)
                loss_time = criterion_time(time_out, time_labels.float())
                loss = loss_act + loss_time
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, act_labels, time_labels = [b.to(device) for b in batch]
                    act_out, time_out = model(inputs)
                    loss_act = criterion_activity(act_out, act_labels)
                    loss_time = criterion_time(time_out, time_labels.float())
                    val_loss += (loss_act + loss_time).item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {loss_time:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        training_time = time.time() - start_time
        model.load_state_dict(best_model)

        # Evaluate
        def evaluate_model(loader):
            model.eval()
            act_preds, act_targets, time_preds, time_targets = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    inputs, act_labels, time_labels = [b.to(device) for b in batch]
                    act_out, time_out = model(inputs)
                    preds = torch.argmax(act_out, dim=1)
                    act_preds.extend(preds.cpu().numpy())
                    act_targets.extend(act_labels.cpu().numpy())
                    time_preds.extend(time_out.cpu().numpy())
                    time_targets.extend(time_labels.cpu().numpy())
            acc = compute_accuracy(act_targets, act_preds)
            f1 = compute_f1(act_targets, act_preds, average='macro')
            mae = compute_mae(time_targets, time_preds)
            return acc, f1, mae

        test_acc, test_f1, test_mae = evaluate_model(test_loader)
        print(f"Run {run} - Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, MAE: {test_mae:.4f}")

        run_results = {
            'run': run,
            'test_activity_accuracy': test_acc,
            'test_activity_f1': test_f1,
            'test_mae': test_mae,
            'training_time_sec': training_time
        }
        results.append(run_results)

    return pd.DataFrame(results)


scenarios = ['scenario_1_A', 'scenario_1_B_75_unique',  'scenario_1_B_40_unique', 'scenario_1_B_20_unique', 'scenario_1_B']
gen_available = [ 'scenario_1_B_75_unique',  'scenario_1_B_40_unique', 'scenario_1_B_20_unique', 'scenario_1_B']

for scenario in scenarios:
    print(f"\n=== Processing {scenario} ===")
    
    file_name = f"dataset/RLRAM_l0.1_s00_{scenario}.csv"
    int_to_activity = pickle.load(open(f"dataset/{scenario}_int_to_activity.p", "rb"))
    activity_to_int = pickle.load(open(f"dataset/{scenario}_activity_to_int.p", "rb"))
    int_to_process = pickle.load(open(f"dataset/{scenario}_int_to_process.p", "rb"))
    process_to_int = pickle.load(open(f"dataset/{scenario}_process_to_int.p", "rb"))

    df = pd.read_csv(file_name, index_col='Unnamed: 0')
    train_df = pd.read_pickle(f'dataset/{scenario}_train_prefix.pkl')
    val_df = pd.read_pickle(f'dataset/{scenario}_val_prefix.pkl')
    test_df = pd.read_pickle(f'dataset/{scenario}_test_prefix.pkl')

    # Load GEN datasets if available
    if scenario in gen_available:
        train_df_gen = pd.read_pickle(f'dataset/{scenario}_train_prefix_gen.pkl')
        val_df_gen = pd.read_pickle(f'dataset/{scenario}_val_prefix_gen.pkl')
        test_df_gen = pd.read_pickle(f'dataset/{scenario}_test_prefix_gen.pkl')

    # Label Encoding for activity predictions
    le = LabelEncoder()
    all_labels = pd.concat([train_df['next_activity_int'], val_df['next_activity_int'], test_df['next_activity_int']])
    le.fit(all_labels)
    train_df['next_activity_int'] = le.transform(train_df['next_activity_int'])
    val_df['next_activity_int'] = le.transform(val_df['next_activity_int'])
    test_df['next_activity_int'] = le.transform(test_df['next_activity_int'])
    evaluation_df = pd.concat([train_df, val_df, test_df])

    if scenario in gen_available:
        train_df_gen['next_activity_int'] = le.transform(train_df_gen['next_activity_int'])
        val_df_gen['next_activity_int'] = le.transform(val_df_gen['next_activity_int'])
        test_df_gen['next_activity_int'] = le.transform(test_df_gen['next_activity_int'])

    # Run baseline
    results_df = run_experiments_pytorch(
        n_runs=10,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        evaluation_df=evaluation_df
    )
    results_df.to_csv(f'results/LSTM_PYTORCH_{scenario}.csv', index=False)

    # Run GEN datasets if available
    if scenario in gen_available:
        results_df_gen = run_experiments_pytorch(
            n_runs=10,
            train_df=train_df_gen,
            val_df=val_df_gen,
            test_df=test_df_gen,
            evaluation_df=evaluation_df
        )
        results_df_gen.to_csv(f'results/LSTM_PYTORCH_{scenario}_GEN.csv', index=False)
