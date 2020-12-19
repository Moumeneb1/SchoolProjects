# -*- coding: utf-8 -*-
"""

@author: barthes
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.matlab as mio
import pomegranate as pg

from duree import duree

    
    
gamma=0.95
alpha=0.65 
beta=0.02 
Nsamples = 100000
# Définition des paramétres du modéle
start_probability = np.array([gamma, 1-gamma])
T = np.array([[1-beta, beta],[1-alpha, alpha]])                # Matrice de transition temporaire

B = np.array([[1,0],[0,1]])               # matrice d'émission temporaire


dicoObs={'sun':0,'rain':1}        # pour transformer les chaines en entier (0 et 1)
dicoState={'sunny':0,'rainy':1}

## Creation de la chaine de Markov
model = pg.HiddenMarkovModel( name="partie 1" )      # Creation instance
# Matrice d'emission
sunny = pg.State( pg.DiscreteDistribution({ 'sun': B[0,0], 'rain':  B[0,1]}), name='sunny' )    # Creation etat beau temps et prob emission
rainy = pg.State( pg.DiscreteDistribution({ 'sun': B[1,0] , 'rain':B[1,1],}), name='rainy' )   # Création de l'état pluie et prob emission
# Matrice de transition
model.add_transitions(model.start,[sunny,rainy],[gamma,1-gamma])  # Probs initiales 
model.add_transitions(sunny, [sunny,rainy], [T[0,0],T[0,1]])     # transitions depuis sunny
model.add_transitions(rainy, [sunny,rainy], [T[1,0],T[1,1]])    # transition depuis rainy

model.add_transition( model.start, sunny, start_probability[0] )       
model.add_transition( model.start, rainy, start_probability[1] )     

model.bake( verbose=True )      # A mettre lorsque la description de la chaine est finalisée


# Génération d'une séquence de longueur Nsamples
obsSeq,statesSeq = model.sample(length=Nsamples,path=True)

obsSeq=np.array([dicoObs[t] for t in obsSeq])      # Conversion des observables String -> int (0:beau, 1 pluie) )
statesSeq=np.array([dicoState[t.name] for t in statesSeq[1:]])   # Conversion des etats String -> int (0:beau, 1 pluie) )

# Chargement des données du fichier RR5MN
ObsMesure = mio.loadmat('RR5MN.mat')['Support'].astype(np.int8).squeeze()

