import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pickle
import time

from framework.utils import compute_mae, find_parallel_sequences

def get_max_activity_id(df):
    max_id = 0
    for prefix in df['prefix_int']:
        if prefix:
            max_id = max(max_id, max(prefix))
    for parallels in df['parallel_sequences']:
        for p in parallels:
            if p['activity_sequence_int']:
                max_id = max(max_id, max(p['activity_sequence_int']))
    return max_id

def set_seed(seed=42):
    random.seed(seed)                      
    np.random.seed(seed)                   
    torch.manual_seed(seed)                
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)       

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

scenarios = ['scenario_1_A', 'scenario_1_B_75_unique',  'scenario_1_B_40_unique', 'scenario_1_B_20_unique', 'scenario_1_B']
gen_available = [ 'scenario_1_B_75_unique',  'scenario_1_B_40_unique', 'scenario_1_B_20_unique', 'scenario_1_B']

for scenario in scenarios:
    file_name=f"dataset/RLRAM_l0.1_s00_{scenario}.csv"
    import pandas as pd
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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
    
    train_df_parallel_sequences = find_parallel_sequences(df_a=train_df, df_b=df[(df['status']!='gateway')], considered_timeframe=100.0, activity_to_int=activity_to_int, process_to_int=process_to_int)#['parallel_sequences'][0]
    val_df_parallel_sequences = find_parallel_sequences(df_a=val_df, df_b=df[(df['status']!='gateway')], considered_timeframe=100.0, activity_to_int=activity_to_int, process_to_int=process_to_int)#['parallel_sequences'][0]
    test_df_parallel_sequences = find_parallel_sequences(df_a=test_df, df_b=df[(df['status']!='gateway')], considered_timeframe=100.0, activity_to_int=activity_to_int, process_to_int=process_to_int)#['parallel_sequences'][0]
    
    if scenario in gen_available:
        train_df_gen_parallel_sequences = find_parallel_sequences(df_a=train_df_gen, df_b=df[(df['status']!='gateway')], considered_timeframe=100.0, activity_to_int=activity_to_int, process_to_int=process_to_int)#['parallel_sequences'][0]
        val_df_gen_parallel_sequences = find_parallel_sequences(df_a=val_df_gen, df_b=df[(df['status']!='gateway')], considered_timeframe=100.0, activity_to_int=activity_to_int, process_to_int=process_to_int)#['parallel_sequences'][0]
        test_df_gen_parallel_sequences = find_parallel_sequences(df_a=test_df_gen, df_b=df[(df['status']!='gateway')], considered_timeframe=100.0, activity_to_int=activity_to_int, process_to_int=process_to_int)#['parallel_sequences'][0]
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import math

    # ---- Model ----
    class RemainingTimePredictionStudent(nn.Module):
        def __init__(self, vocab_size, max_seq_len, d_model=36, nhead=4, ff_dim=64, dropout=0.2):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

            self.norm = nn.LayerNorm(d_model)
            #self.pool = nn.AdaptiveAvgPool1d(1)

            self.lupi_fc = nn.Linear(d_model, 64)
            self.lupi_agg_fc = nn.Linear(64, 64)
            self.concat_fc = nn.Linear(d_model + 64, 30)
            self.output_layer = nn.Linear(30, 1)
            self.dropout = nn.Dropout(dropout)

        def encode_sequence(self, seq, t):
            B, T = seq.shape
            seq_embed = self.embedding(seq)
            #time_embed = self.sinusoidal_time_encoding(t, self.d_model)
            x = seq_embed #+ time_embed

            pad_mask = (seq == 0)
            x = x.transpose(0, 1)  # (T, B, d_model)
            x = self.encoder(x, src_key_padding_mask=pad_mask)
            x = self.norm(x)
            x = x.transpose(0, 1)  # (B, T, d_model)

            # Get the index of the last non-padding token for each sequence
            lengths = (seq != 0).sum(dim=1) - 1  # shape: (B,)

            # Gather the final relevant token from each sequence
            last_token_repr = x[torch.arange(B), lengths]  # shape: (B, d_model)
            return last_token_repr


        def forward(self, x_main, t_main, x_par, t_par, par_mask):
            main_enc = self.encode_sequence(x_main, t_main)

            B, N, T = x_par.shape
            x_par = x_par.view(B * N, T)
            t_par = t_par.view(B * N, T)

            par_enc = self.encode_sequence(x_par, t_par)
            lupi_vec = F.relu(self.lupi_fc(par_enc))
            lupi_vec = lupi_vec.view(B, N, -1)

            par_mask = par_mask.unsqueeze(-1)
            lupi_vec = lupi_vec * par_mask

            sum_vec = lupi_vec.sum(dim=1)
            valid_counts = par_mask.sum(dim=1).clamp(min=1)
            lupi_agg = sum_vec / valid_counts
            lupi_agg = F.relu(self.lupi_agg_fc(lupi_agg))

            combined = torch.cat([main_enc, lupi_agg], dim=1)
            x = self.dropout(F.relu(self.concat_fc(combined)))
            return self.output_layer(x).squeeze(1)


    # ---- Dataset ----
    class FastLUPIDataset(Dataset):
        def __init__(self, df, max_seq_len, max_parallel=10):
            self.df = df
            self.max_seq_len = max_seq_len
            self.max_parallel = max_parallel

        def pad(self, seq, max_len, pad_value=0.0):
            return seq[-max_len:] + [pad_value] * max(0, max_len - len(seq))

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]

            x_main = torch.tensor(self.pad(row['prefix_int'], self.max_seq_len, 0), dtype=torch.long)
            t_main = torch.tensor(self.pad(row['list_timestamps_case'], self.max_seq_len, 0.0), dtype=torch.float32)
            y = torch.tensor(row['remaining_time'], dtype=torch.float32)

            parallel_x, parallel_t, par_mask = [], [], []
            for p in row['parallel_sequences'][:self.max_parallel]:
                parallel_x.append(self.pad(p['activity_sequence_int'], self.max_seq_len, 0))
                parallel_t.append(self.pad(p['timestamps_case_related'], self.max_seq_len, 0.0))
                par_mask.append(1)

            while len(parallel_x) < self.max_parallel:
                parallel_x.append([0] * self.max_seq_len)
                parallel_t.append([0.0] * self.max_seq_len)
                par_mask.append(0)

            return (
                x_main,
                t_main,
                torch.tensor(parallel_x, dtype=torch.long),
                torch.tensor(parallel_t, dtype=torch.float32),
                y,
                torch.tensor(par_mask, dtype=torch.bool)
            )

    # --- MAE Calculation ---
    def compute_mae(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred)).item()

    # --- Train Function ---
    # ---- Training ----
    def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=100, patience=10):
        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for x_main, t_main, x_par, t_par, y, par_mask in train_loader:
                x_main, t_main = x_main.to(device), t_main.to(device)
                x_par, t_par, y = x_par.to(device), t_par.to(device), y.to(device)
                par_mask = par_mask.to(device)

                optimizer.zero_grad()
                preds = model(x_main, t_main, x_par, t_par, par_mask)
                loss = loss_fn(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_main, t_main, x_par, t_par, y, par_mask in val_loader:
                    x_main, t_main = x_main.to(device), t_main.to(device)
                    x_par, t_par, y = x_par.to(device), t_par.to(device), y.to(device)
                    par_mask = par_mask.to(device)

                    preds = model(x_main, t_main, x_par, t_par, par_mask)
                    val_loss += loss_fn(preds, y).item()

            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping.")
                    break

        model.load_state_dict(best_model)
        return model

    

    def run_experiments(n_runs, train_df_parallel_sequences, val_df_parallel_sequences, test_df_parallel_sequences, save_path_prefix="models/busch2025_model4_run.pt"):
        results = []
        for run in range(1, n_runs + 1):
            print(f"\n=== Run {run} ===")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            max_seq_len = 50#, 50 for scenario 1
            max_parallel = 10
            batch_size = 128
            d_model = 36
            num_layers = 1
            num_epochs = 200
            patience = 20

            #vocab_size = max(max(seq) for seq in train_df_parallel_sequences['prefix_int']) + 1
            vocab_size = get_max_activity_id(train_df_parallel_sequences) + 1


            train_dataset = FastLUPIDataset(train_df_parallel_sequences, max_seq_len, max_parallel)
            val_dataset = FastLUPIDataset(val_df_parallel_sequences, max_seq_len, max_parallel)
            test_dataset = FastLUPIDataset(test_df_parallel_sequences, max_seq_len, max_parallel)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = RemainingTimePredictionStudent(vocab_size=vocab_size, max_seq_len=max_seq_len, d_model=d_model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.L1Loss()

            start_time = time.time()
            model = train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, patience)

            training_time = time.time() - start_time

            run_results = {"run": run, "training_time_sec": training_time}
            
            torch.save(model.state_dict(), save_path_prefix)
            model.load_state_dict(torch.load(save_path_prefix))
            model.eval()

            for dataset in [{'name': 'Test', 'data': test_loader}]:
                

                all_preds, all_targets = [], []
                with torch.no_grad():
                    for batch in dataset['data']:
                        x_main, t_main, x_par, t_par, y, par_mask = [i.to(device) for i in batch] 
                        x_main, t_main = x_main.to(device), t_main.to(device)
                        x_par, t_par, y = x_par.to(device), t_par.to(device), y.to(device)
                        par_mask = par_mask.to(device)

                        preds = model(x_main, t_main, x_par, t_par, par_mask)
                        all_preds.append(preds.cpu())
                        all_targets.append(y.cpu())

                preds_cat = torch.cat(all_preds)
                targets_cat = torch.cat(all_targets)
                mae = compute_mae(targets_cat, preds_cat)
                print(f"{dataset['name']} Remaining Time MAE: {mae:.4f}")
                run_results[f"test_mae"] = mae

            results.append(run_results)

        return pd.DataFrame(results)



    results = run_experiments(
        n_runs=10,
        train_df_parallel_sequences=train_df_parallel_sequences,
        val_df_parallel_sequences=val_df_parallel_sequences,
        test_df_parallel_sequences=test_df_parallel_sequences
    )
    # results
    # results.mean()
    # results
    results.to_csv(f'results/MODEL4_{scenario}.csv')
    ### apply on GEN dataset
    if scenario in gen_available:
        results = run_experiments(
            n_runs=10,
            train_df_parallel_sequences=train_df_gen_parallel_sequences,
            val_df_parallel_sequences=val_df_gen_parallel_sequences,
            test_df_parallel_sequences=test_df_gen_parallel_sequences
        )
        # results.mean()
        # results
        results.to_csv(f'results/MODEL4_{scenario}_GEN.csv')

    