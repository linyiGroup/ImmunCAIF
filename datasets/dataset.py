from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import pickle


import sys 
sys.path.append("../") 
from utils.embedding import Embedder


class PHLADataset(Dataset):
    def __init__(self, csv_path, config):
        self.data = pd.read_csv(csv_path)
        self.peptideList = self.data.peptide.tolist()
        self.HLAList = self.data.HLA.tolist()
        self.labelList = self.data.label.tolist()

        self.esmcEmbeddingPath = config["esmcEmbeddingPath"]
        self.esmcEmbeddingDict = None
        with open(config["esmcEmbeddingDictPath"], "rb") as f:
            self.esmcEmbeddingDict = pickle.load(f)

        self.config = config
    def __len__(self):
        return len(self.labelList)
        # return 10

    def __getitem__(self, idx):

        peptide = self.peptideList[idx]
        hla = self.HLAList[idx]
        label = self.labelList[idx]
        
        embeddingName = self.esmcEmbeddingDict[peptide]
        peptideESM = np.load(self.esmcEmbeddingPath + embeddingName)

        embeddingName = self.esmcEmbeddingDict[hla]
        hlaESM = np.load(self.esmcEmbeddingPath + embeddingName)
        

        context = {
            "label": label,
            "peptide": {
                "embedding": torch.tensor(peptideESM, dtype=torch.float32),
                "length": len(peptide)
            },
            "hla": {
                "embedding": torch.tensor(hlaESM, dtype=torch.float32),
                "length": len(hla)
            }
        }
        return context
    


class PTCRDataset(Dataset):
    def __init__(self, csv_path, config):
        self.data = pd.read_csv(csv_path)
        self.peptideList = self.data.peptide.tolist()
        self.tcrList = self.data.tcr.tolist()
        self.labelList = self.data.label.tolist()

        self.esmcEmbeddingPath = config["esmcEmbeddingPath"]
        self.esmcEmbeddingDict = None
        with open(config["esmcEmbeddingDictPath"], "rb") as f:
            self.esmcEmbeddingDict = pickle.load(f)

        self.config = config
    def __len__(self):
        return len(self.labelList)
        # return 10

    def __getitem__(self, idx):

        peptide = self.peptideList[idx]
        tcr = self.tcrList[idx]
        label = self.labelList[idx]
        
        embeddingName = self.esmcEmbeddingDict[peptide]
        peptideESM = np.load(self.esmcEmbeddingPath + embeddingName)

        embeddingName = self.esmcEmbeddingDict[tcr]
        hlaESM = np.load(self.esmcEmbeddingPath + embeddingName)
        

        context = {
            "label": label,
            "peptide": {
                "embedding": torch.tensor(peptideESM, dtype=torch.float32),
                "length": len(peptide)
            },
            "tcr": {
                "embedding": torch.tensor(hlaESM, dtype=torch.float32),
                "length": len(tcr)
            }
        }
        return context
    
