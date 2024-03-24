import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from train import train_loop
from model import Architecture_Pixel,ResidualBlock_CNN


def load_show_mnist( ):
    """
    Charge le jeu de données MNIST.
    normalise les données

    Returns:
    trainloader: DataLoader pour les données d'entraînement.
    testloader: DataLoader pour les données de test.
    trainset: Dataset d'entraînement.
    testset: Dataset de test.
    """
    MEAN_MNIST = (0.1307,)
    STD_MNIST = (0.3081,)

    # Pour transformer nos donnees en Tensor et les Normaliser
    transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_MNIST, STD_MNIST)])

    #On divise nos donnees en Train et Test puis on les transforme en Dataloader
    batch_size = 16 #Taille qu'ils utilisent dans l'article (5.2)

    trainset = torchvision.datasets.MNIST(root='./data', train=True, # On telecharge nos donnees Train
                                            download=True, transform=transform_mnist)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, # On telecharge nos donnees Test
                                        download=True, transform=transform_mnist)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    st.write('Affichage des images :')
    #affiche les trois premières images
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axes[i].imshow(torch.permute(trainset[i][0], (1, 2, 0)), cmap='gray' )
        
    st.pyplot(fig)

    return(trainloader,testloader,trainset,testset,MEAN_MNIST,STD_MNIST)


def load_show_cifar10():
    """
    Charge le jeu de données CIFAR-10.
    normalise les données
    Affiche les trois premières images

    Returns:
    trainloader: DataLoader pour les données d'entraînement.
    testloader: DataLoader pour les données de test.
    trainset: Dataset d'entraînement.
    testset: Dataset de test.
    """

    # Pour transformer nos donnees en Tensor et les Normaliser
    transform_CIFAR10 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616))])

    #On divise nos donnees en Train et Test puis on les transforme en Dataloader
    batch_size2 = 16

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, # On telecharge nos donnees Train
                                            download=True, transform=transform_CIFAR10 )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size2,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, # On telecharge nos donnees Test
                                        download=True, transform=transform_CIFAR10 )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size2,
                                            shuffle=False, num_workers=2)
    
    st.write('Affichage des images :')
    # Les valeurs des pixels sont normalisées, nous devons les dénormaliser pour obtenir les valeurs originales (0-255)
    denorm = transforms.Normalize(mean = [-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616], std = [1/0.2470, 1/0.2435, 1/0.2616])
    #affiche les trois premières images
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in range(3):
        axes[i].imshow(torch.permute(denorm(trainset[i][0]), (1, 2, 0)))
        
    st.pyplot(fig)
    return(trainloader,testloader,trainset,testset,[-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616],[1/0.2470, 1/0.2435, 1/0.2616])


def afficher_page_accueil():
    
    
    st.title("Bienvenue dans notre projet de modèle génératif d'image")
    st.write(' Nous explorerons le modèle Pixel CNN sur les jeux de données MNIST et CIFAR-10.')



def afficher_choix_jeux_de_donnees():
    global trainloader, testloader, trainset, testset, mean, std,dataset

    st.title('Choix du jeu de données et affichage d\'images')

    # Sélection du jeu de données
    dataset = st.radio('Choisissez un jeu de données :', ('MNIST', 'CIFAR-10'))

    if dataset == 'MNIST':
        st.write('Chargement du jeu de données MNIST...')
        trainloader,testloader,trainset,testset,mean,std = load_show_mnist()

    elif dataset == 'CIFAR-10':
        st.write('Chargement du jeu de données CIFAR-10...')
        trainloader,testloader,trainset,testset,mean,std = load_show_cifar10()
    


def afficher_train_page_modele(device):
    global trainloader, testloader, trainset, testset, mean, std,dataset
    
    st.title('Pixel CNN')
    

    st.write('la Loss utilisé est la negative log likelihood')
    st.write("Optimizer = Adam")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam

    if dataset == 'MNIST' : 
        in_channels = 1 
        out_channels = 256
        file_weight = 'PixelCNN_MNIST.pth'
    else : 
        in_channels = 3 
        out_channels = 3*256
        file_weight = 'PixelCNN_CIFAR-10.pth'

    # Afficher les valeurs des variables globales
    st.write(f'Vous avez choisi le jeu de données {dataset}')

    # Options pour l'utilisateur
    option = st.radio("Choisissez une option :", ( "Entraîner un nouveau modèle","Utiliser un modèle pré-entraîné"))
    if option == "Entraîner un nouveau modèle" :

        # Interface pour définir les paramètres d'entraînement
        epochs = st.slider("Nombre d'époques :", min_value=1, max_value=100, value=10, step=1)
        h = st.slider("Nombre de neurones :", min_value=1, max_value=128, value=5, step=1)
        lr = st.slider("Taux d'apprentissage :", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
        p = st.slider("Nombre de blocs résiduels : ", min_value=1, max_value=8, value=2, step=1)

        go =st.checkbox("Commencer l'entraînement")

        if go:
            start_time = time.time()
            model = train_loop(Architecture_Pixel, in_channels, out_channels, h, ResidualBlock_CNN, nn.LogSoftmax(), p, optimizer, criterion, device, lr, trainloader, epochs,mean,std )
            
            st.write(f"Training with {device} lasts: {np.round((time.time()-start_time)/60,2)} minutes\n")
            # Enregistrer l'architecture et les poids du modèle
            torch.save(model.state_dict(), 'user.pth')
            #Create an instance of the class
            model = Architecture_Pixel(in_channels, out_channels, h, ResidualBlock_CNN, nn.Softmax(dim=0)).to(device = device)
            
            model.load_state_dict(torch.load('user.pth'), map_location=torch.device(device)) #import model weights


        
            st.title('Visualisation de nos résultats')
            l1, l2 = viz_im(model,p,device)
            display_images_in_line(l1,'image réelle')
            display_images_in_line(l2,'image prédite')
            

    else :
        #modèle déjà entraîner
        if dataset == 'MNIST' :
            st.write('Voici les paramètre pour du modèle')
            st.write("Nombre d'époques = 10")
            st.write("Nombre Nombre de neurones = 5")
            st.write("Learning rate = 0.09 ")
            st.write("Nombre de blocs résiduels = 3 ")
            h = 5
            p = 3
        else :
            st.write('Voici les paramètre pour du modèle')
            st.write("Nombre d'époques = 20")
            st.write("Nombre Nombre de neurones = 5")
            st.write("Learning rate = 0.07 ")
            st.write("Nombre de blocs résiduels = 3 ")
            h = 5
            p = 3
        #Create an instance of the class
        model = Architecture_Pixel(in_channels, out_channels, h, ResidualBlock_CNN, nn.Softmax(dim=0)).to(device )
        model.load_state_dict(torch.load(file_weight, map_location=torch.device(device)))  #import model weights
        
        st.title('Visualisation de nos résultats sur le testset')
        #affiche des images du testest et les prédictions 
        l1, l2 = viz_im(model,p,device)
        display_images_in_line(l1,'image réelle')
        display_images_in_line(l2,'image prédite')
    
def viz_im(model, p, device):
   """
      Creates and returns 2 lists of images from a testloader: one containing real images, the other containing model predictions.

      Parameters
      ----------
      model : nn.Module
         The trained model.   
      p : int
         Number of residual blocks.
      device : string
         The device to use.
      
      Returns
      -------
      list1 : tensor
         List with the true values of each pixel for 4 random testloader images.
      list2 : tensor
         List with predictions for each pixel in 4 random testloader images.
   """
   global trainloader, testloader, trainset, testset, mean, std, dataset

   denorm = transforms.Normalize(mean = [-0.4915/0.2470, -0.4823/0.2435, -0.4468/0.2616], std = [1/0.2470, 1/0.2435, 1/0.2616])
   list1 = []
   list2 = []
   rand = torch.randint(0, 625, (4,))
   for i in rand: 
        image, _ = testset[i]

        #Real image
        if dataset == 'MNIST' : 
            image_true = np.ravel(image.numpy())
            image_true = image_true.reshape(28, 28)

            list1.append(image_true)

            #Prediction
            y = model(image.to(device), p)
            y = torch.reshape(y, (256, 784)) #reshape to get a 1d vector, but it still has the 256 channels

            y_pred = np.zeros(784) #our future image

            for i in range(784):
                probs = y[:, i]
                y_pred[i] = torch.multinomial(probs, 1)

            image_hat = y_pred.reshape(28,28)
            list2.append(image_hat)
        else:
            image_true = (denorm(image)*255).to(torch.int)#Pixel values are normalized, so we need to denormalize them to obtain the original values (0-255)
            image_true = image_true.permute(1, 2, 0)
            list1.append(image_true)

            #Prediction
            y = model(image.to(device), p)
            y = torch.reshape(y, (256, 3, 1024)) #reshape to get a 1d vector, but it still has the 256 channels

            yr_pred = np.zeros(1024) #Channels RGB
            yg_pred = np.zeros(1024)
            yb_pred = np.zeros(1024)

            for i in range(1024):
                probs_r = y[:, 0, i]
                yr_pred[i] = torch.multinomial(probs_r, 1)
                probs_g = y[:, 1, i]
                yg_pred[i] = torch.multinomial(probs_g, 1)
                probs_b = y[:, 2, i]
                yb_pred[i] = torch.multinomial(probs_b, 1)

            y_pred = torch.stack([torch.tensor(yr_pred),torch.tensor(yg_pred) ,torch.tensor(yb_pred)]).to(torch.int)

            image_hat = torch.reshape(y_pred, (3, 32, 32))
            image_hat = image_hat.permute(1, 2, 0)
            list2.append(image_hat)
     

   return(list1, list2)



def display_images_in_line(image_list, title):
   """
      Displays images from a list.

      Parameters
      ----------
      image_list : list
         A list of images to view.
      title : string
         The title of the image to view.
   """
   fig = plt.figure(figsize=(len(image_list)*2, 2))
   for i, image_tensor in enumerate(image_list):
      
      plt.subplot(1, len(image_list), i+1)
      plt.imshow(image_tensor.squeeze(), cmap = 'gray')
      plt.title(title)
      plt.axis('off')
   st.pyplot(fig)
    
                 