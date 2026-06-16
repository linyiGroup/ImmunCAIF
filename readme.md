

# (ImmunCAIF) CAIF: Cross-Adaptive Attention for Multi-Source Feature Fusion in Unified Prediction of Antigen Binding to HLA and TCR

This repository contains the official implementation of **ImmunCAIF**, a deep learning framework for accurate prediction of peptide binding to HLA molecules and T-cell receptors (TCRs).

---

## 📂 Project Structure

```text
ImmunCAIF/
├── 01_ESMcEmbeddingGeneration.py       # Precompute ESM-based embeddings for sequences
├── 02_train_evaluation_pHLA.py         # Train & evaluate on antigen-HLA binding task
├── 02_train_evaluation_pTCR.py          # Train & evaluate on antigen-TCR binding task
├── 03_evaluation_pHLA.py                # Standalone evaluation for antigen-HLA models
├── 03_evaluation_pTCR.py                # Standalone evaluation for antigen-TCR models
├── configs/
│   ├── config_model.yaml               # Core model configuration
│   ├── config_train_pHLA.yaml          # Training hyperparameters (pHLA)
│   ├── config_train_pTCR.yaml          # Training hyperparameters (pTCR)
│   ├── config_5_fold_evaluation_pHLA.yaml  # 5-fold cross-validation config (pHLA)
│   ├── config_5_fold_evaluation_pTCR.yaml  # 5-fold cross-validation config (pTCR)
│   └── ESMcEmbedding.yaml              # Configuration for ESM embedding generation
├── datasets/
│   └── dataset.py                      # Data loading
├── models/
│   ├── ESMc/                           # ESM-based protein embedding modules
│   ├── final_model_state/              # Saved model checkpoints and training states
│   └── models.py                       # Implementation of the ImmunCAIF model
├── utils/
│   ├── embedding.py                    # Embedding-related helper functions
│   ├── ESMcEmbedding.py                # ESM embedding wrapper and utilities
│   ├── optimization.py                 # Optimizer, learning rate scheduler, etc.
│   └── utils.py                        # General utilities (metrics, logging, etc.)
└── requirements.txt                    # Python dependencies
```

##  🚀 Environment Setup
Install required packages:
bash

```
pip install -r requirements.txt
```

## 🧬 Step 1: Generate ESM-based Embeddings

Precompute embeddings for HLA, TCR, and peptide sequences using the ESM language model:
bash

```
python 01_ESMcEmbeddingGeneration.py 
```


## 🏋️ Step 2: Train the Model

Train on pHLA binding
bash

```
python 02_train_evaluation_pHLA.py 
```

Train on pTCR binding
bash

```
python 02_train_evaluation_pTCR.py 
```


## 📊 Step 3: Evaluate the Model
Evaluate pHLA binding prediction

(Officially released model state is available in the `models/final_model_state` directory)

bash

```
python 03_evaluation_pHLA.py 
```

Evaluate TCR-peptide binding prediction

bash
```
python 03_evaluation_pTCR.py 
```


## 📄 Citation
The paper has not been published
<!-- If you find this code useful in your research, please cite our work:
```plaintext
@article{ImmunCAIF,
  title={CAIF: Cross-Adaptive Attention for Accurate HLA and TCR-Peptide Binding Prediction},
  author={[Your Name]},
  journal={[Journal Name]},
  year={[Year]}
}
``` -->
