from util import  afficher_page_accueil, afficher_choix_jeux_de_donnees, afficher_train_page_modele, afficher_visualisation
import streamlit as st
import numpy as np
import torch



def main() :
    # Initialisation des variables globales
    trainloader = None
    testloader = None
    trainset = None
    testset = None
    mean = None
    std = None
    dataset = None
    model = None
    p=None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    st.sidebar.title('Navigation')
    # Ajoutez les différentes sections de votre application à la barre latérale
    section = st.sidebar.radio('Sections', ('Accueil',  'Modèle'))

    if section == 'Accueil':
        afficher_page_accueil()
        afficher_choix_jeux_de_donnees()

    elif section == 'Modèle':
        afficher_train_page_modele(device)
        
        
    

if __name__ == "__main__":
    main()

        