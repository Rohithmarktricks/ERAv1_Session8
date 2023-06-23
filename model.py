import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class GroupNormModel(nn.Module):
    def __init__(self, num_classes):
        super(GroupNormModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=1, stride=1, padding=0)

        # Replace with Group Normalization
        self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.group_norm1 = nn.GroupNorm(4, 16)  # Apply Group Normalization after conv4

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)

        self.conv7 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)

        # Replace with Group Normalization
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(8, 32)  # Apply Group Normalization after conv8

        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final convolutional layer for classification
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Residual connections
        self.residual1 = nn.Conv2d(12, 12, kernel_size=1, stride=1, padding=0)
        self.residual2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
#         self.residual3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.relu(self.conv3(out))
        out = self.pool1(out)
        
        out = nn.functional.relu(self.conv4(out))
        out = self.group_norm1(out)  # Apply Group Normalization
        out = nn.functional.relu(out)

        out = nn.functional.relu(self.conv5(out))
        out = nn.functional.relu(self.conv6(out))
        
        out = nn.functional.relu(self.conv7(out))
        out = self.pool2(out)
        out = nn.functional.relu(self.conv8(out))
        out = self.group_norm2(out)  # Apply Group Normalization
        out = nn.functional.relu(out)

        out = nn.functional.relu(self.conv9(out))
        out = nn.functional.relu(self.conv10(out))

        out = self.global_pool(out)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)

        return out


class LayerNormModel(nn.Module):
    def __init__(self, num_classes):
        super(LayerNormModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=1, stride=1, padding=0)

        # Replace with Layer Normalization
        self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.layer_norm1 = nn.LayerNorm([16, 16, 16])  # Apply Layer Normalization after conv4

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)

        # Replace with Layer Normalization
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.layer_norm2 = nn.LayerNorm([16, 8, 8])  # Apply Layer Normalization after conv8

        self.conv9 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)


        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final convolutional layer for classification
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Residual connections
        self.residual1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        self.residual2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
#         self.residual3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.relu(self.conv3(out))
        out = self.pool1(out)
        
        out = nn.functional.relu(self.conv4(out))

        out = self.layer_norm1(out)  # Apply Layer Normalization
        out = nn.functional.relu(out)

        out = nn.functional.relu(self.conv5(out))
        out = nn.functional.relu(self.conv6(out))
        out = self.pool2(out)
        
        out = nn.functional.relu(self.conv7(out))
        out = nn.functional.relu(self.conv8(out))

        out = self.layer_norm2(out)  # Apply Layer Normalization
        out = nn.functional.relu(out)

        out = nn.functional.relu(self.conv9(out))
        out = nn.functional.relu(self.conv10(out))

        
        out = self.global_pool(out)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)

        return out

class BatchNormModel(nn.Module):
    def __init__(self, num_classes):
        super(BatchNormModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 12, kernel_size=1, stride=1, padding=0)

        # Replace with Batch Normalization
        self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(16)  # Apply Batch Normalization after conv4

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)

        self.conv7 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)

        # Replace with Batch Normalization
        self.conv8 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(32)  # Apply Batch Normalization after conv8

        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)

        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final convolutional layer for classification
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        
        # Residual connections
        self.residual1 = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0)
        self.residual2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
#         self.residual3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.relu(self.conv2(out))
        out = nn.functional.relu(self.conv3(out))
        
        out = self.pool1(out)
#         print(out.shape)
#         residual = self.residual1(out)
        
        
        out = nn.functional.relu(self.conv4(out))
        out = self.batch_norm1(out)  # Apply Batch Normalization
        out = nn.functional.relu(out)

        out = nn.functional.relu(self.conv5(out))
        out = nn.functional.relu(self.conv6(out))

        out = nn.functional.relu(self.conv7(out))
        out = self.pool2(out)
        
        out = nn.functional.relu(self.conv8(out))
        out = self.batch_norm2(out)  # Apply Batch Normalization
        out = nn.functional.relu(out)

        out = nn.functional.relu(self.conv9(out))
        out = nn.functional.relu(self.conv10(out))

        
        out = self.global_pool(out)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)

        return out



def get_model(normalization_type):
    '''Function that takes the type of normalization technique that has to be applied 
    and returns the model.'''
    if normalization_type == "BN":
        return BatchNormModel(num_classes=10)
    elif normalization_type == "LN":
        return LayerNormModel(num_classes=10)
    elif normalization_type == "GN":
        return GroupNormModel(num_classes=10)
    else:
        raise ValueError("Normlization has to be one of BN/LN/GN")