#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:38:57 2018

@author: barthes
Version 1.0
"""
import numpy as np
import matplotlib.pyplot as plt
from TpHmmUtilit import GaussianHMM, Words
from sklearn.metrics import confusion_matrix

if __name__ =='__main__' :
    winlen=0.02         # taille d'une frame en seconde
    winstep=0.01        # decalage temporel entre une frame et la suivante en seconde
    highfreq=4000       # frequence max en Hz
    lowfreq=0           # fréquence min en Hz
    nfilt=26            # Nonbre de filtres calculés
    numcep=12           # Nombre de coefficients de Mel
    nfft=256            # Nombre de points pour le calcul du spectre
    
    methode='mfcc'      # Choix de la feature (sprectrum, filter ou mfcc)
    label='apple'       # mot choisi 
    Fx=0                # Composante en abcisse pour affichage
    Fy=1                # composante en ordonnée pour affichage
    featStart=0         # Choix de la composante min de feature
    featStop=4          # Choix de la composante max de feature
    Nstates=3         # Nombre d'état de la chaine de Markov
    
    # lecture des fichiers audio et calcul des features
    words=Words(rep='audio',name='audio',numcep=numcep,lowfreq=lowfreq,
                highfreq=None,winlen=winlen,winstep=winstep,nfilt=nfilt,nfft=nfft)  
    
    # On extrait une liste avec les 15 enregistrements de 'apple' 
    #en utilisant la méthode mfcc
    liste=words.getFeatList(label=label,methode=methode,featStart=featStart,featStop=featStop)  
    
    # On crée et on entraine un HMM avec la liste précédente (composée 
    # de 15 enregistrements de apple)
    Model=GaussianHMM(liste=liste, Nstates=Nstates)     
    
    # Affiche tous les individus du mot apple dans le plan Fx, Fy ainsi que 
    #les ellipes à 95% associées à chaque états
    
    Model.plotGaussianConfidenceEllipse(words,Fx=Fx,Fy=Fy,color='b')    
    
    # On calcule la log densité de probabilité pour chacun des 15 enregistrements 
    # de apple par l'algorithme Forward
    logprobs=Model.log_prob(liste)      
    print('Log densite de probabilité des {} enregistrements de {}:\n{}\n'.format(len(liste),label,logprobs))
   
    
    # prediction des séquence d'état optimale par l'algorithme de 
    # Viterbi pour chacun des 15 enregistrements
    predictedStates=Model.predict(liste)       
    for i,l in enumerate(predictedStates):
        print('Séquence des Etats optimaux enregistrement {}:\n {}'.format(i, l))
    
    # Visualisation des parametres de la chaine de Markov
    np.set_printoptions(precision=2,suppress=True)
    print('Matrice de transition :\n{}'.format(Model.getTrans()))
    print('Prob initiale : \n{}'.format(Model.getPi0()))
    for i in range(Nstates):
        print('\nEtats {} :'.format(i))
        print('cov:\n{}'.format(Model.getCov()[i]))    
        print('Mu:\n{}'.format(Model.getMu()[i]))
    
