import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np


class MaskedConv(nn.Conv2d):

	"""
		The masked convolution function.

		Parameters
		----------
		mask_type : string, "A" ou "B"
				The type of mask to be used for convolution.
		in_channels : int
				Number of input channels, 1 (black & white) or 3 (RGB).
		out_channels : int
				The number of classes in our multinomial distribution (256), the value a pixel can take.

		Source
		------
		https://github.com/AlexanderMath/pixcnn
    """

	def __init__(self, mask_type, *args, **kwargs):
		super(MaskedConv, self).__init__(*args, **kwargs)

		self.mask_type = mask_type #type de masque utilisé
		assert mask_type in ['A', 'B'], "Unknown Mask Type" #check if mask is valid (must be 'A' or 'B')

		self.register_buffer('mask', self.weight.data.clone()) #mask initialization by cloning convolution layer weight data

		_, depth, height, width = self.weight.size()
		self.mask.fill_(1) #initializes kernel weights to 1

		if mask_type == 'A':
			#Set to 0 the cells after our pixel xi (include)
			self.mask[:, :, height//2, width//2:] = 0
			self.mask[:, :, height//2+1:, :] = 0
		else: #Mask B
			#Set to 0 the cells after our pixel xi (exclude)
			self.mask[:, :, height//2, width//2+1:] = 0
			self.mask[:, :, height//2+1:, :] = 0


	def forward(self, x):
		self.weight.data *= self.mask #the weights of the convolution layer are modified by multiplying them element by element with the mask
		return super(MaskedConv, self).forward(x) #perform convolution operation with modified weights




class ResidualBlock_CNN(nn.Module):
    """
      Our residual block for the PixelCNN model.

      Parameters
      ----------
      h : int
          Number of channels in the layers.

      Attributes
      ----------
      conv : MaskedConv
          A masked convolution
      relu : nn.ReLU
          ReLU layers

      Returns
      -------
      torch.add(inputs, out) : tensor
          Sum of input and output
    """

    def __init__(self, h):
        super(ResidualBlock_CNN, self).__init__()
        self.conv1 = MaskedConv(mask_type = 'B', in_channels = 2*h, out_channels = h, kernel_size = 1)
        self.conv2 = MaskedConv(mask_type = 'B', in_channels = h, out_channels = h, kernel_size = 3, padding = 1) #to get the right image size
        self.conv3 = MaskedConv(mask_type = 'B', in_channels = h, out_channels = 2*h, kernel_size = 1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        out = nn.ReLU()(x)
        return torch.add(inputs, out) #sum of input and output
	




class Architecture_Pixel(nn.Module):
    """
      The general architecture of our models.

      Parameters
      ----------
      in_channels : int
          Number of input channels, 1 (black & white) or 3 (RGB).
      out_channels : int
          The number of classes in our multinomial distribution (256), the value a pixel can take.
      h : int
          Number of channels in the layers.
      residual_block : function
          The residual block to be used in our architecture, which defines the model.
      end_activation : torch.nn, default = nn.LogSoftmax()
          The activation function to be used at the end of our architecture.
      x: tensor
          The input tensor, of shape (batch_size, C, n, n), depending on the dataset.
          For MNIST : batch-size = 512, C = 1, n = 28
          For CIFAR-10 : batch-size = 256, C = 3, n = 32
      p : int
          Number of blocks.
          5 if PixelCNN
          6 if PixelRNN

      Attributes
      ----------
      conv : MaskedConv
          A masked convolution
      relu : nn.ReLU
          ReLU layers

      Returns
      -------
      out : tensor
          Vector of probabilities for each class(out_channels).
    """

    def __init__(self, input_channels, output_channels, h, residual_block, end_activation = nn.LogSoftmax()):
        super(Architecture_Pixel, self).__init__()

        self.conv1 = MaskedConv(mask_type = 'A', in_channels = input_channels, out_channels = 2*h, kernel_size = 7, padding = 3)
        self.block = residual_block(h)
        self.conv2 = MaskedConv(mask_type = 'B', in_channels = 2*h, out_channels = 2*h, kernel_size = 1)
        self.conv3 = MaskedConv(mask_type = 'B', in_channels = 2*h, out_channels = output_channels, kernel_size = 1)
        self.relu = nn.ReLU()
        self.activation = end_activation

    def forward(self, x, p):
        x = self.conv1(x)

        for i in range(p):
          x = self.block(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        out = self.activation(x)
        return out
    



def train_loop(architecture, input_channels, output_channels,  h, residual_block, end_activation, p, optimizer, criterion, device, lr, trainloader, n_epochs = 10):
    """
        Function to train our models.

        Parameters
        ----------
        architecture : function
            The architecture of the neural network to be trained
        in_channels : int
            Number of input channels, 1 (black & white) or 3 (RGB).
        out_channels : int
            The number of classes in our multinomial distribution (256), the value a pixel can take.
        h : int
            Number of channels in the layers.
        residual_block : function
            The residual block to be used in our architecture, which defines the model.
        end_activation : torch.nn
            The activation function to be used at the end of our architecture.
        p : int
            Number of residual blocks.
            5 if PixelCNN
            6 if PixelRNN
        optimizer : torch.optim.optimizer.Optimizer
            The optimizer to use
        criterion : nn.modules._Loss
            The loss function to minimize
        device : string
            The device to use.
        lr : float
            The learning rate
        trainloader : torch.utils.data.DataLoader
            The train DataLoader
        n_epochs : int, default = 10
            Number of epochs to do

        Returns
        -------
        model : nn.Module
            The trained model.
    """
    #Build the model
    model = architecture(input_channels, output_channels, h, residual_block, end_activation).to(device = device)
    optimizer = optimizer(model.parameters(), lr = lr)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (x,y) in enumerate(trainloader):
        # assign a device for the inputs
            inputs = x.to(device)
      
            if input_channels == 1 :
                label = torch.round((x.to(device) * STD_MNIST[0] + MEAN_MNIST[0]) * 255) #denormalize to obtain original labels (0-255)
                labels = torch.reshape(label, (16, 1, 28, 28)).squeeze().long()
            else :
                denorm = transforms.Normalize(mean = [-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616], std = [1/0.2470, 1/0.2435, 1/0.2616])
                label = torch.round(denorm(x.to(device))*255).to(torch.int) #denormalize to obtain original labels (0-255)
                labels = label.long()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs, p = p) # dim = (16,256,28,28) or (16, 3*256, 32, 32), we need to go back to 3d tensor for the loss function
            loss = criterion(outputs, labels)

            # backward
            loss.backward()

            # optimize
            optimizer.step()

            running_loss += loss.item()


    print(f'Epoch {epoch + 1} / {n_epochs} | Loss: {running_loss / len(trainloader)}')
    running_loss = 0.0
  return model