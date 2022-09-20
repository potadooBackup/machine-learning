import numpy as np
import torch 

from src.hyper import ProblemType, PROBLEM_TYPE

def same_seed(seed: np.uint):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_feature(train_data, valid_data, test_data, select_all = True):
    if select_all:
        return train_data, valid_data, test_data
    else:
        pass

# Training function.
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0

    for data in data_loader:
        counter += 1

        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(features)
        # Calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Backpropagation
        loss.backward()
        # Update the optimizer parameters
        optimizer.step()

        # Calculate the accuracy
        if PROBLEM_TYPE == ProblemType.CLASSIFICATION:
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = (
        100. * (train_running_correct / len(data_loader.dataset))        
    ) if PROBLEM_TYPE == ProblemType.CLASSIFICATION else None

    return epoch_loss, epoch_acc

# Validation function.
def validate(model, data_loader, criterion, device):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for data in data_loader:
            counter += 1
            
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(features)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            # Calculate the accuracy.
            if PROBLEM_TYPE == ProblemType.CLASSIFICATION:
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = (
        100. * (valid_running_correct / len(data_loader.dataset))
    ) if PROBLEM_TYPE == ProblemType.CLASSIFICATION else None

    return epoch_loss, epoch_acc

def predict(model, test_loader, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in test_loader:
        x = x.to(device)                        
        with torch.no_grad():                   
            outputs = model(x.float())
            if PROBLEM_TYPE == ProblemType.CLASSIFICATION:
                _, outputs = torch.max(outputs.data, 1)
            preds.append(outputs.detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    return preds