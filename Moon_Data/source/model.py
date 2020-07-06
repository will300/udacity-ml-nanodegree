import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()
        
        # define all layers, here
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.3)
        self.hidden_layer = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Sigmoid()
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        # your code, here
        x = self.input_layer(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        x = self.hidden_layer(x)
        x = self.output(x)
        
        return x