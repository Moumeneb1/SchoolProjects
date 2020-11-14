import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

def plot_event(indices,lien=""):
    serie = loadmat(lien+'serie_temporelle.mat')['serie']
    fourchettes = loadmat(lien+'fourchettes.mat')['fourchettes']
    if(type(indices)==int):
        plt.plot(serie[fourchettes[indices,0]-1:fourchettes[indices,1]])
        plt.title("série temporelle de l'événement {0}".format(str(indices+1)),
                  fontsize=20)
        plt.xlabel("Temps [min]",fontsize=20)
        plt.ylabel("RR [mm/h]",fontsize=20)
        plt.grid()
        normaliser_axes(18)
    elif(type(indices)==list):
        nb_c = 1
        nb_l = int(np.ceil(len(indices)/nb_c))
        for j,i in enumerate(indices):
            plt.subplot(nb_l,nb_c,j+1)
            plt.plot(serie[fourchettes[i,0]-1:fourchettes[i,1]])
            plt.title("série temporelle de l'événement {0}".format(str(i+1)),
                      fontsize=20)
            plt.xlabel("Temps [min]",fontsize=20)
            plt.ylabel("RR [mm/h]",fontsize=20)
            plt.grid()
            normaliser_axes(18)

def normaliser_axes(taille):
    for tickLabel in plt.gca().get_xaxis().get_ticklabels():
        tickLabel.set_fontsize(taille)
    for tickLabel in plt.gca().get_yaxis().get_ticklabels():
        tickLabel.set_fontsize(taille)
        
