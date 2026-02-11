# Runtime Adaptive Token Pruning for LLM Based Biomedical Transformers

This project demonstrates fine-tuning **BioBERT** for Named Entity Recognition (NER) in the biomedical domain (e.g., detecting diseases and chemicals). It further explores advanced **Adaptive Inference** techniques—including rule-based and learned (MLP) token pruning and recovery—to optimize model efficiency without compromising significantly on accuracy.

## Project Overview

- **Task**: Biomedical Named Entity Recognition (NER).
- **Model**: BioBERT (`dmis-lab/biobert-base-cased-v1.1`).
- **Datasets**: BC5CDR (Chemical/Disease).
- **Key Innovation**: Adaptive token pruning using attention entropy and an MLP-based controller, with a memory buffer for token recovery to maintain high F1 scores.

---

## Notebook Structure & Cell Breakdown

The `AI_&_ML_Lab_Project.ipynb` notebook is organized into the following logical sections:

### 1. Environment Setup & Prerequisites
*   **Package Installation**: Installs core libraries including `transformers`, `datasets`, `seqeval`, `gradio`, and `accelerate`.
*   **GPU Verification**: Checks for CUDA availability (optimized for Tesla T4 found in Google Colab).
*   **Drive Integration**: Mounts Google Drive for persistent storage of datasets and trained models.

### 2. Data Loading & Preprocessing
*   **Dataset Handling**: Functions to load PubTator format files (`.rel` or `.txt`) and parse them into token-level annotations.
*   **BIO Tagging**: Implements logic to convert character-level entity offsets into token-level BIO (Begin, Inside, Outside) tags compatible with BERT sub-tokens.
*   **Hugging Face Integration**: Converts processed data into `Datasets` objects for efficient batching and pipeline integration.

### 3. Baseline Model Training
*   **Fine-tuning**: Standard training of a `BertForTokenClassification` model (BioBERT) on the biomedical NER data.
*   **Evaluation**: Uses `seqeval` to calculate entity-level metrics (Precision, Recall, F1).
*   **Serialization**: Saves the baseline model weights and tokenizer to Google Drive.

### 4. Adaptive Inference: Rule-Based Pruning
*   **Token Importance**: Calculates the **Attention Entropy** of each token across BERT layers to identify "unimportant" tokens.
*   **Adaptive Controller**: A rule-based controller (`controller_get_keep_mask`) that prunes tokens based on entropy thresholds and model confidence.
*   **Memory Buffer**: Implements a `MemoryBuffer` class to store embeddings of pruned tokens, allowing the model to recall them if an "Entity Loss" or "Entropy Spike" is detected.

### 5. Learned Controller: MLP-Based Pruning
*   **Data Preparation**: Extracts features (attention distributions, hidden states, confidence) from the baseline model to train a lighter controller.
*   **MLP Architecture**: Defines `LearnedController`, a simple Multi-Layer Perceptron trained to predict token importance.
*   **Adaptive Pipeline**: Integrates the MLP controller into the inference flow to make faster, more dynamic pruning decisions.

### 6. Comprehensive Evaluation & Results
*   **Latency Benchmarking**: Measures the real-world inference speedup on GPU.
*   **FLOPs Estimation**: Quantifies the computational savings (reduction in Floating Point Operations) achieved via token pruning.
*   **Ablation Studies**: Analyzes the impact of different hyper-parameters (Lambda factor, Min Keep Ratio) and the effectiveness of the Recovery Mechanism.
*   **Visualization**: Generates Matplotlib charts comparing Baseline, Rule-Based, and MLP Adaptive models in terms of F1 vs. Latency.
---

# Adaptive Token Pruning for Efficient NER

This repository implements an **Adaptive Dynamic Token Pruning** framework for Named Entity Recognition (NER) using BERT. By identifying and removing redundant tokens during inference, the system achieves significant speedups in computational throughput (FLOPs) while maintaining high entity-extraction accuracy through a novel **Memory Buffer & Recovery** mechanism.

## Key Features

* **Multi-Strategy Controllers:** Includes Rule-based, MLP-based, and Entropy-driven pruning logic.
* **Dynamic Recovery System:** Automatically detects low-confidence predictions and "recovers" tokens from a memory buffer to restore context.
* **Efficiency Focused:** Optimized for reduction in quadratic  self-attention complexity.
* **Comprehensive Evaluation:** Built-in tools for measuring latency, Error Analysis, Fairness Analysis, Ablation Studies, FLOPs.

---

## System Architecture

The project moves beyond static model compression by making real-time decisions per sequence:

1. **Partial Inference:** The first  layers process the full sequence.
2. **Importance Scoring:** Tokens are ranked based on attention weights and classification entropy.
3. **Dynamic Pruning:** Non-essential tokens are masked out to accelerate the remaining layers.
4. **Confidence Check:** If the pruned output is uncertain, the **Recovery Trigger** re-inserts tokens for a second, high-fidelity pass.

---

## Performance Summary (Example Results)

| Model Configuration | F1 Score | Token Reduction | Theoretical Spe |
| --- | --- | --- | --- |
| **Baseline BERT** | 0.923 | 0.0% | 0.0% |
| **Rule-Based Pruning** | 0.916 | 30% | 51% |
| **MLP Adaptive + Recovery** | 0.923 | 2-4% | 4-9% |

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/spk-22/Bio-PruneNER.git


# Install dependencies
pip install -r requirements.txt

# Run Streamlit Dashboard
streamlit run streamlit_ner_demo.py
```

---

## Ablation Studies

The project includes scripts to test the sensitivity of:

* **Lambda Factor:** Balancing aggressive pruning vs. accuracy.
* **Min Keep Ratio:** Ensuring a safety floor for the number of tokens processed.
