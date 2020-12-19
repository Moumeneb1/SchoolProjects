# -*- coding: utf-8 -*-
"""

@author: barthes
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.matlab as mio
import pomegranate as pg

from duree import duree

Nsamples = 100000
# Définition des paramétres du modéle

start_probability = np.array([0.5, 0.5])
T = np.array([[0.6,0.4],[0.3,0.7]])                # Matrice de transition temporaire

B = np.array([[0.1,0.4,0.5],[0.6,0.3,0.1]])        # matrice d'émission temporaire


dicoObs={'fine': 0 ,'moyenne':1, 'epaisse':2}        # pour transformer les chaines en entier (0,1 et 2)
dicoState={'cold':0 ,'hot':1}

## Creation de la chaine de Markov
model = pg.HiddenMarkovModel( name="partie 3" )      # Creation instance
# Matrice d'emission

# Creation etat beau temps et prob emission
cold = pg.State( pg.DiscreteDistribution({ 'fine': B[0,0],'moyenne': B[0,1],'epaisse':B[0,2]}), name='cold' )


# Creation etat beau temps et prob emission
hot = pg.State( pg.DiscreteDistribution({ 'fine': B[1,0],'moyenne': B[1,1],'epaisse':B[1,2]}), name='hot' )


# Matrice de transition
model.add_transitions(model.start,[cold,hot],[0.5, 0.5])  # Probs initiales 
model.add_transitions(cold, [cold,hot],[T[0,0],T[0,1]])     # transitions depuis sunny
model.add_transitions(hot,  [cold,hot],[T[1,0],T[1,1]])     # transition depuis rainy


model.add_transition( model.start, cold, start_probability[0] )       
model.add_transition( model.start, hot, start_probability[1] )   

model.bake( verbose=True )      # A mettre lorsque la description de la chaine est finalisée


# Génération d'une séquence de longueur Nsamples
obsSeq,statesSeq = model.sample(length=Nsamples,path=True)

obsSeq=np.array([dicoObs[t] for t in obsSeq])      # Conversion des observables String -> int (0:beau, 1 pluie) )
statesSeq=np.array([dicoState[t.name] for t in statesSeq[1:]])   # Conversion des etats String -> int (0:beau, 1 pluie) )

# Chargement des données du fichier RR5MN
ObsMesure = mio.loadmat('RR5MN.mat')['Support'].astype(np.int8).squeeze()

