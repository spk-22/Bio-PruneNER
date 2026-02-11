# AI & ML Lab Project: Adaptive Inference for Biomedical NER

This project demonstrates fine-tuning **BioBERT** for Named Entity Recognition (NER) in the biomedical domain (e.g., detecting diseases and chemicals). It further explores advanced **Adaptive Inference** techniques—including rule-based and learned (MLP) token pruning and recovery—to optimize model efficiency without compromising significantly on accuracy.

## Project Overview

- **Task**: Biomedical Named Entity Recognition (NER).
- **Model**: BioBERT (`dmis-lab/biobert-base-cased-v1.1`).
- **Datasets**: BC5CDR (Chemical/Disease) and NCBI Disease.
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

### 7. Interactive Demo
*   **Gradio App**: Provides a user-friendly web interface. Users can input biomedical text, and the app visualizes detected entities using the adaptive inference pipeline.

---

## Key Findings
- **Efficiency**: The adaptive pruning methods achieve significant FLOPs reduction (e.g., ~30-50%) and latency improvements.
- **Accuracy**: The recovery mechanism ensures that critical entity tokens are restored, keeping the F1 score close to the baseline model.
- **Scalability**: The modularized code allows for testing different buffer sizes and trigger strategies for various deployment constraints.
