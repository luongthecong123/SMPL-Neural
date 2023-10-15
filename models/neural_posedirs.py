import torch
import torch.nn as nn
from torch.nn.utils import prune

# Define the input and output dimensions
input_dim = 207 # The shape of the input
hidden_dim_1 = 184
hidden_dim_2 = 69 # The shape of the hidden layer
output_dim = 20670 # The shape of the output

# Define the dropout rate
dropout_rate = 0.4 # The dropout rate for the hidden layer

# Define the neural network class

class Neural_posedirs(nn.Module):
    def __init__(self):
        super(Neural_posedirs, self).__init__()
        # self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_1 = nn.Linear(input_dim, hidden_dim_2)
        self.relu = nn.LeakyReLU(negative_slope=0.01)    
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(hidden_dim_2, output_dim)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.layer_2(x)
        return x

# class Neural_posedirs(nn.Module):
#     def __init__(self):
#         super(Neural_posedirs, self).__init__()
#         self.layer_1 = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         x = self.layer_1(x)
#         return x

# # Define the neural network class
# class Neural_posedirs(nn.Module):
#     def __init__(self):
#         super(Neural_posedirs, self).__init__()
#         self.dropout1 = nn.Dropout(dropout_rate)
#         self.layer_1 = nn.Linear(input_dim, hidden_dim_1)  # Try Leaky RELU maybe ?
#         self.layer_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
#         self.dropout2 = nn.Dropout(dropout_rate)
#         self.relu = nn.LeakyReLU(negative_slope=0.01)
#         # Define the dropout layer        
        
#         # Define the second layer
#         self.dropout3 = nn.Dropout(dropout_rate)
#         self.layer_3 = nn.Linear(hidden_dim_2, output_dim)

#     def forward(self, x):
#         x = self.dropout1(x)
#         x = self.layer_1(x)
#         x = self.relu(x)
#         x = self.dropout2(x)
#         x = self.layer_2(x)
#         x = self.relu(x)
#         x = self.dropout3(x)
#         x = self.layer_3(x)
#         return x
    
# from torchviz import make_dot

# net = Neural_posedirs()
# x = torch.randn(207)
# y = net(x)
# make_dot(y, params=dict(net.named_parameters()))

if __name__ == "__main__":
    net = Neural_posedirs()

    input_vert = torch.zeros([64, 20670])
    input_rot = torch.zeros([64, 207])
    output = net(input_rot)
    
    prune.l1_unstructured(net.layer_1, name='weight', amount=0.3)
    prune.l1_unstructured(net.layer_1, name='bias', amount=0.3)
    prune.l1_unstructured(net.layer_2, name='weight', amount=0.3)
    prune.l1_unstructured(net.layer_2, name='bias', amount=0.3)    

    print(list(net.layer_1.named_parameters()))
    print(list(net.layer_1.named_buffers()))



