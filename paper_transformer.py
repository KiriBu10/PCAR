import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
import pandas as pd
from framework.utils import compute_mae, compute_accuracy, compute_f1

def set_seed(seed=42):
    random.seed(seed)                      
    np.random.seed(seed)                   
    torch.manual_seed(seed)                
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)       

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

scenarios = [
    # 'scenario_1_A', 
    #          'scenario_1_B_75_unique',  
             'scenario_1_B_40_unique', 
            #  'scenario_1_B_20_unique', 
            #  'scenario_1_B', 
             'bpi2020_2processes_massive_share']
gen_available = [ 
    # 'scenario_1_A', 
    # 'scenario_1_B_75_unique', 
      'scenario_1_B_40_unique', 
    #   'scenario_1_B_20_unique', 
    #   'scenario_1_B',
      'bpi2020_2processes_massive_share']

for scenario in scenarios:
    file_name=f"dataset/RLRAM_l0.5_s00_{scenario}.csv"

    # Parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_seq_len = 20
    d_model = 36
    batch_size = 32
    num_epochs = 100
    patience = 10
    int_to_activity = pickle.load( open( f"dataset/{scenario}_int_to_activity.p", "rb" ) )
    activity_to_int = pickle.load( open( f"dataset/{scenario}_activity_to_int.p", "rb" ) )

    int_to_process = pickle.load( open( f"dataset/{scenario}_int_to_process.p", "rb" ) )
    process_to_int = pickle.load( open( f"dataset/{scenario}_process_to_int.p", "rb" ) )

    df = pd.read_csv(file_name, index_col='Unnamed: 0')
    train_df = pd.read_pickle(f'dataset/{scenario}_train_prefix.pkl')
    val_df = pd.read_pickle(f'dataset/{scenario}_val_prefix.pkl')
    test_df = pd.read_pickle(f'dataset/{scenario}_test_prefix.pkl')
    # gen dataset
    if scenario in gen_available:
        train_df_gen = pd.read_pickle(f'dataset/{scenario}_train_prefix_gen.pkl')
        val_df_gen = pd.read_pickle(f'dataset/{scenario}_val_prefix_gen.pkl')
        test_df_gen = pd.read_pickle(f'dataset/{scenario}_test_prefix_gen.pkl')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    all_labels = pd.concat([train_df['next_activity_int'], val_df['next_activity_int'], test_df['next_activity_int']])
    le.fit(all_labels)
    train_df['next_activity_int'] = le.transform(train_df['next_activity_int'])
    val_df['next_activity_int'] = le.transform(val_df['next_activity_int'])
    test_df['next_activity_int'] = le.transform(test_df['next_activity_int'])
    evaluation_df = pd.concat([train_df, val_df, test_df])

    # gen dataset
    if scenario in gen_available:
        train_df_gen['next_activity_int'] = le.transform(train_df_gen['next_activity_int'])
        val_df_gen['next_activity_int'] = le.transform(val_df_gen['next_activity_int'])
        test_df_gen['next_activity_int'] = le.transform(test_df_gen['next_activity_int'])
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import time


    # Positional Encoding
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super(PositionalEncoding, self).__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1), :]

    # RemainingTimeTransformer
    class RemainingTimeTransformer(nn.Module):
        def __init__(self, vocab_size, max_seq_len, d_model=36, nhead=4, ff_dim=64, dropout=0.1, num_temporal_features=1):
            super(RemainingTimeTransformer, self).__init__()
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.global_pool = nn.AdaptiveAvgPool1d(1)

            self.temporal_fc = nn.Linear(num_temporal_features, 32)
            self.dropout1 = nn.Dropout(dropout)
            self.concat_fc = nn.Linear(d_model + 32, 128)
            self.dropout2 = nn.Dropout(dropout)
            self.output_layer = nn.Linear(128, 1)

        def forward(self, x, temporal_features):
            x = self.embedding(x)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)
            x = self.transformer_encoder(x)
            x = x.transpose(0, 1).permute(0, 2, 1)
            x = self.global_pool(x).squeeze(-1)

            temporal_x = F.relu(self.temporal_fc(temporal_features))
            x = torch.cat([x, temporal_x], dim=1)
            x = self.dropout1(F.relu(self.concat_fc(x)))
            x = self.dropout2(x)
            return self.output_layer(x).squeeze(1)

    # Dataset uses prefix time
    class ProcessDataset(Dataset):
        def __init__(self, df, max_seq_len):
            self.sequences = df['prefix_int'].tolist()
            self.remaining_times = df['remaining_time'].tolist()
            self.prefix_times = df['timestamp_case'].tolist()
            self.max_seq_len = max_seq_len

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            seq = self.sequences[idx]
            remaining_time = self.remaining_times[idx]
            prefix_time = self.prefix_times[idx]

            padded = seq[:self.max_seq_len] + [0] * (self.max_seq_len - len(seq))
            padded = torch.tensor(padded, dtype=torch.long)
            temporal = torch.tensor([prefix_time], dtype=torch.float32)

            return padded, temporal, torch.tensor(remaining_time, dtype=torch.float32)
    # Training loop with early stopping
    def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=100, patience=10):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for x, temporal, y in train_loader:
                x, temporal, y = x.to(device), temporal.to(device), y.to(device)
                optimizer.zero_grad()
                preds = model(x, temporal)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, temporal, y in val_loader:
                    x, temporal, y = x.to(device), temporal.to(device), y.to(device)
                    preds = model(x, temporal)
                    loss = loss_fn(preds, y)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        model.load_state_dict(best_model_state)
        return model
    def run_experiments(n_runs, train_df, val_df, test_df, save_path_prefix="models/bukhsh2021_transformer_run"):
        results = []

        for run in range(1, n_runs + 1):
            print(f"\n=== Run {run} ===")

            # Build vocab_size fresh in case data changes
            vocab_size = max(max(seq) for seq in train_df['prefix_int']) + 1

            # Datasets
            train_dataset = ProcessDataset(train_df, max_seq_len)
            val_dataset = ProcessDataset(val_df, max_seq_len)
            test_dataset = ProcessDataset(test_df, max_seq_len)

            # Loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            # Model
            model = RemainingTimeTransformer(
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                d_model=d_model,
                num_temporal_features=1
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.L1Loss()

            # Train
            start_time = time.time()
            model = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=num_epochs, patience=patience)
            training_time = time.time() - start_time
            # Save
            model_path = f"{save_path_prefix}_{run}.pt"
            torch.save(model.state_dict(), model_path)

            # Evaluate
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            model.eval()

            run_results = {"run": run, "training_time_sec": training_time}

            for dataset in [{'name':'Test','data': test_loader}]:
                all_preds, all_targets = [], []

                with torch.no_grad():
                    for x, temporal, y in dataset['data']:
                        x, temporal = x.to(device), temporal.to(device)
                        preds = model(x, temporal)
                        all_preds.append(preds.cpu())
                        all_targets.append(y)

                time_preds = torch.cat(all_preds)
                time_targets = torch.cat(all_targets)

                mae = compute_mae(time_targets, time_preds)
                print(f"{dataset['name']} Remaining Time MAE: {mae:.4f}")
                run_results[f"test_mae"] = mae

            results.append(run_results)

        return pd.DataFrame(results)

    results = run_experiments(
        n_runs=10,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )
    # results.mean()
    # results
    results.to_csv(f'results/TRANSFORMER_{scenario}.csv')
    ### apply on GEN dataset
    if scenario in gen_available:
        results = run_experiments(
            n_runs=10,
            train_df=train_df_gen,
            val_df=val_df_gen,
            test_df=test_df_gen
        )
        # results.mean()
        # results
        results.to_csv(f'results/TRANSFORMER_{scenario}_GEN.csv')