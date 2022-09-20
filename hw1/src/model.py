import os

# tuning
from ray import tune

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.hyper import const_config, ProblemType, PROBLEM_TYPE
from src.utils import train, validate
from src.data import load_dataset, train_valid_split, get_data_loaders

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x:torch.Tensor = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x


def train_and_validate(config):
    # Load dataset
    dataset = load_dataset('./covid.train.csv')

    # Get training and validation dataset
    train_dataset, valid_dataset = train_valid_split(dataset, 0.2, const_config['random_seed'])

    # Get training and validation data loaders,
    # ignore test data loader for now.
    train_loader, valid_loader, _ = get_data_loaders(train_dataset, valid_dataset,
                                                     batch_size = config['batch_size'])

    # Initialize the model
    model = My_Model(dataset.shape[1]-1).to(const_config['device'])

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(
        model.parameters(), lr=config['lr'], momentum=0.9
    )

    # start the training
    for epoch in range(const_config['n_epochs']):
        # print(f"[INFO]: Epoch {epoch+1} of {const_config['n_epochs']}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, const_config['device']
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, const_config['device']
        )
  
        # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_loss:.3f}")
        # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_loss:.3f}")
        # print('-'*50)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        if PROBLEM_TYPE == ProblemType.CLASSIFICATION:
            tune.report(
                loss=valid_epoch_loss, accuracy=valid_epoch_acc
            )
        elif PROBLEM_TYPE == ProblemType.REGRESSION:
            tune.report(
                loss=valid_epoch_loss
            )
    
    # torch.save((model.state_dict(), optimizer.state_dict()), path)