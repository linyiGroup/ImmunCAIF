from utils.utils import load_config, Logger, log_format
from datasets.dataset import PTCRDataset
from torch.utils.data import DataLoader
from models.models import PTCRModel
from utils.optimization import train_one_epoch, validate

from torch import nn
import torch.optim as optim
import torch
import pandas as pd
import numpy as np
    


def train_valid_pTCR(config, config_model):

    logger = Logger(config["logPath"] + "train_pTCR.log")

    train_file_name = "train_set.csv"
    val_file_name = "test_set.csv"

    train_set = PTCRDataset(csv_path=config["dataPath"]+train_file_name, config=config)
    val_set = PTCRDataset(csv_path=config["dataPath"]+val_file_name, config=config)

    train_loader = DataLoader(train_set, batch_size=config["batchSize"], shuffle=True, num_workers=config["numWorker"])
    val_loader = DataLoader(val_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])

    external_set = PTCRDataset(csv_path=config["dataPath"]+"external_test_set.csv", config=config)
    external_loader = DataLoader(external_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])

    model = PTCRModel(config_model).to(config["device"])
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learningRate"]))

    logger.write("Start training...\n\n")
    for epoch in range(config["epochs"]):
        train_performance = train_one_epoch(model, train_loader, loss_function, optimizer, config["device"])
        val_performance = validate(model, val_loader, loss_function, config["device"])
        external_performance = validate(model, external_loader, loss_function, config["device"])

        msg = log_format(train_performance, 
                         test_performance=val_performance, 
                         external_performance=external_performance,
                         epoch=epoch)
        logger.write(msg)

        torch.save(model.state_dict(), config["modelStatePath"] + "pTCR_model.pth")
        
    model.load_state_dict(torch.load(config["modelStatePath"] + "pTCR_model.pth"))
    
    test_set = PTCRDataset(csv_path=config["dataPath"]+"test_set.csv", config=config)
    test_loader = DataLoader(test_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    test_performance = validate(model, test_loader, loss_function, config["device"])
    test_performance = pd.DataFrame([test_performance], index=["test"])

    external_set = PTCRDataset(csv_path=config["dataPath"]+"external_test_set.csv", config=config)
    external_loader = DataLoader(external_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    external_performance = validate(model, external_loader, loss_function, config["device"])
    external_performance = pd.DataFrame([external_performance], index=["external test"])

    return test_performance, external_performance


if __name__ == "__main__":

    config = load_config("./configs/config_train_pTCR.yaml")
    config_model = load_config("./configs/config_model.yaml")

    test_performance, external_performance = train_valid_pTCR(config, config_model)
        
    all_performance = pd.concat([test_performance, external_performance])
    all_performance = pd.DataFrame(all_performance).round(4)
    all_performance = all_performance[["roc_auc", "pr_auc", "accuracy", "f1", "mcc"]]
    all_performance.to_csv(config["logPath"] + "performance_pTCR.csv")


