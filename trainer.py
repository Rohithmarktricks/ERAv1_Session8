import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer:
    '''Trainer class for Layer & Group Normalization'''
    def __init__(self, model):
        self.model = model
        self.train_losses = []
        self.train_acc = []
        self.test_losses = []
        self.test_acc = []

    def train(self, device, train_loader, optimizer, epoch):
        self.model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = self.model(data)

            # Calculate loss
            loss = F.nll_loss(y_pred, target)
            self.train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)


    def test(self, device, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        idx = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        self.test_acc.append(100. * correct / len(test_loader.dataset))

    def get_stats(self):
        return self.train_losses, self.train_acc, self.test_losses, self.test_acc


def get_misclassified_images(model, device, test_loader):
    '''Takes the test_loader as input and returns 
    the dictionary{index: [image, true_label, predicted_label]} 
    of misclassfied images.
    '''
    misclassified_imgs = {}
    model.eval()
    idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output  = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for sample in range(data.shape[0]):
                if (target[sample] != pred[sample]):
                    misclassified_imgs[idx] = [data[sample].cpu(), target[sample].cpu(), pred[sample].cpu()]
                idx += 1
    return misclassified_imgs