from utils.utils import load_config
from datasets.dataset import PTCRDataset
from torch.utils.data import DataLoader
from models.models import PTCRModel
from utils.optimization import validate

import torch
import pandas as pd
    


def valid_pTCR(config, config_model):
    model = PTCRModel(config_model)
    model.load_state_dict(torch.load("./models/final_model_state/pHLA_model.pth" + "pTCR_model.pth"))
    model.to(config["device"])
    
    test_set = PTCRDataset(csv_path=config["dataPath"]+"test_set.csv", config=config)
    test_loader = DataLoader(test_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    test_performance = validate(model, test_loader, None, config["device"])
    test_performance = pd.DataFrame([test_performance], index=["test"])

    external_set = PTCRDataset(csv_path=config["dataPath"]+"external_test_set.csv", config=config)
    external_loader = DataLoader(external_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    external_performance = validate(model, external_loader, None, config["device"])
    external_performance = pd.DataFrame([external_performance], index=["external test"])

    return test_performance, external_performance


if __name__ == "__main__":

    config = load_config("./configs/config_train_pTCR.yaml")
    config_model = load_config("./configs/config_model.yaml")

    test_performance, external_performance = valid_pTCR(config, config_model)
        
    all_performance = pd.concat([test_performance, external_performance])
    all_performance = pd.DataFrame(all_performance)
    all_performance = all_performance[["roc_auc", "pr_auc", "accuracy", "f1", "mcc"]] * 100
    all_performance = all_performance.round(2)
    all_performance.to_csv(config["logPath"] + "performance_pTCR.csv")


