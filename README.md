# PCAR: Parallel Context-Aware Remaining Time Prediction in Contention-Intensive Processes
Remaining time prediction for ongoing process instances is a core task in process mining. 
Existing approaches typically focus on the internal execution history of a single case, assuming that process instances execute in isolation.
However, in real-world environments, multiple cases often run concurrently and interact through shared resources or external dependencies. 
This contention, where the execution of one case influences another, can significantly affect case duration. 
Traditional models that consider only the intra-case history fail to capture these contextual signals, resulting in suboptimal predictions in environments characterized by rich inter-instance dependencies.
To address this limitation, we propose **PCAR**, a context-aware method for remaining time prediction that leverages event logs to model both intra-case behavior and inter-case context. Specifically, our method employs dual Transformer encoders to capture both effects.
Empirical evaluations on five different scenarios demonstrate that PCAR outperforms state-of-the-art methods, particularly in **high-contention scenarios**, yielding more accurate and robust predictions.

---
Authors:


---
PCAR considers:
- **Intra-case behavior**: the internal activity sequence of a target process instance.
- **Inter-case context**: influence from concurrently active (parallel) cases.

---

## Method Overview

PCAR uses a dual-encoder neural architecture:
- A **target case encoder** processes the history of the current case.
- A **parallel case encoder** processes a fixed number of temporally overlapping cases.
- Representations from both encoders are fused and passed to a prediction head.

The model is implemented using **Transformer encoders** and trained using standard event log data.



## Project Structure

The repository is organized into datasets, preprocessing utilities, model implementations, experiment scripts, and result analysis notebooks.

```text
.
├── dataset/                         # Preprocessed datasets used for training and evaluation
├── dataset_xes/                     # Event logs in XES format
├── framework/                       # Shared framework code, utilities, and helper modules
├── raw_dataset/                     # Raw downloaded event logs before preprocessing
├── results/                         # Saved experiment outputs, logs, and evaluation results
│
├── .gitignore                       # Git ignore rules
├── 4TU-links.yaml                   # Links/metadata for datasets from the 4TU repository
├── downloaded-datasets.yaml         # Record of downloaded datasets and sources
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
│
├── config_train.py                  # Training configuration and shared experiment settings
├── data_aquisition.py               # Dataset download/acquisition script
│
├── pcar.py                          # Core PCAR implementation for simulation-based experiments
├── pcar_rw.py                       # Core PCAR implementation for real-world experiments
│
├── paper_pcar.py                    # Main PCAR experiment script (simulation scenarios)
├── paper_pcar_rw.py                 # Main PCAR experiment script (real-world event logs)
├── paper_transformer.py             # ProcessTransformer baseline experiments (simulation)
├── paper_transformer_rw.py          # ProcessTransformer baseline experiments (real-world)
├── paper_lstm_pytorch.py            # LSTM baseline experiments (simulation)
├── paper_lstm_pytorch_rw.py         # LSTM baseline experiments (real-world)
├── amiri2024.py                     # PGTNet baseline experiments (simulation)
├── amiri2024_rw.py                  # PGTNet baseline experiments (real-world)
├── senderovich2017.ipynb            # Senderovich et al. baseline / exploratory notebook
├── senderovich2017_rw.py            # Senderovich et al. baseline for real-world logs
│
├── preprocessing.ipynb             # Preprocessing pipeline for simulated datasets
├── preprocessing_real_world.ipynb  # Preprocessing pipeline for real-world datasets
├── paper_results.ipynb             # Result aggregation, analysis, and figure generation             

```
---

## Installation

**Requirements:**
- Python 3.11.10

**Install dependencies:**

```bash
pip install -r requirements.txt
```

## Contact

For inquiries or support, please contact:
- **Email**: [***]
