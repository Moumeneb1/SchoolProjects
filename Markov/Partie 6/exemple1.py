#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:38:57 2018
Modified 11 Dec 2019
@author: barthes
Version 1.1
"""
import numpy as np
import matplotlib.pyplot as plt
from TpHmmUtilit import GaussianHMM, Words, autocorr
from sklearn.metrics import confusion_matrix

if __name__ =='__main__' :
    winlen=0.02         # taille d'une frame en seconde
    winstep=0.01        # decalage temporel entre une frame et la suivante en seconde
    nfft=256            # Méthode Spectrum : Nombre de points pour le calcul de la FFT => spectre nfft/2 + 1 valeurs
    nfilt=26            # Méthode filter : Nonbre de filtres calculés 
    numcep=12           # Méthode mfcc Nombre de coefficients de Mel
       
    
    # lecture de tous les fichiers audio contenu dans le réprtoire 'audio' et calcul des features
    words=Words(rep='audio',name='exemple 1',numcep=numcep,winlen=winlen,winstep=winstep,nfilt=nfilt,nfft=nfft,filterLow=False)  
    
    # On récupère la liste des mots disponibles et on l'affiche
    listeDesMotsDisponibles = words.getLabels()     
    print('Les mots disponibles sont :\n{}'.format(listeDesMotsDisponibles))
    
    # Affiche les features du mots 'apple' record 0 pour les 3 méthodes
    label='apple'      # mot choisi 
    record = 0       # Selectionne pour le mot choisi un enregistrement parmi les 15 (0 .. 14)
    
    words.plotOneWord(label=label,num=record)       
    
    # Pour les 3 méthodes on affiche les histogrammes des différentes features Fi du mot apple
    # Affiche tous les individus du mot apple dans le plan des Features Fx, Fy
    methodes=['spectrum','filter','mfcc']       # Liste des méthodes disponibles
    Fx=0                # Composante en abcisse pour affichage
    Fy=1                # composante en ordonnée pour affichage
    
    featStart=0         # premiere feature à considérer
    featStop=11         # derniere feature à considérer

    # Affichage des histogrammes et des features dans un plan Fx, Fy pourles 3 méthodes
    for methode in methodes:    # On boucle sur les trois méthodes et on affiche
        words.histFeatures(label,methode,featStart,featStop)  # affiche les histogrammes des composantes Fi de tous les mots 'apple' 
    
    
    for methode in methodes:    # On boucle sur les trois méthodes et on affiche
        words.plotFeatureXY(label,methode,Fx,Fy)   # Affiche pour tous les mots 'apple' les composantes Fx=0 et Fy=1
    