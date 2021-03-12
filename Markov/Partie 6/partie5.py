# -*- coding: utf-8 -*-
"""

@author: barthes
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.matlab as mio
import pomegranate as pg

from duree import duree

Nsamples = 100
# Définition des paramétres du modéle

start_probability = np.array([1, 0, 0])
T = np.array([[0.5 , 0.4 , 0.1],[0.3 , 0.4 , 0.3 ],[0.1 , 0.2 , 0.7 ]])                # Matrice de transition temporaire

B = np.array([[0.5 , 0.5],[0.25,0.75], [0.75, 0.25]])        # matrice d'émission temporaire


dicoObs={'pile': 0 ,'face':1}        # pour transformer les chaines en entier (0,1 et 2)
dicoState={'P1':0 ,'P2':1, 'P3':2}

## Creation de la chaine de Markov
model = pg.HiddenMarkovModel( name="partie 5" )      # Creation instance
# Matrice d'emission

# Creation etat beau temps et prob emission
p1 = pg.State( pg.DiscreteDistribution({ 'pile': B[0,0],'face': B[0,1]}), name='P1' )

p2 = pg.State( pg.DiscreteDistribution({ 'pile': B[1,0],'face': B[1,1]}), name='P2' )

p3 = pg.State( pg.DiscreteDistribution({ 'pile': B[2,0],'face': B[2,1]}), name='P3')




# Matrice de transition
model.add_transitions(model.start,[p1,p2,p3],[1, 0, 0])  # Probs initiales 
model.add_transitions(p1,  [p1,p2,p3],[T[0,0],T[0,1],T[0,2]])     # transitions depuis sunny
model.add_transitions(p2,  [p1,p2,p3],[T[1,0],T[1,1],T[1,2]])     # transition depuis rainy
model.add_transitions(p3,  [p1,p2,p3],[T[2,0],T[2,1],T[2,2]])     # transition depuis rainy


model.add_transition( model.start, p1, start_probability[0] )       
model.add_transition( model.start, p2, start_probability[1])   
model.add_transition( model.start, p3, start_probability[2])  

model.bake( verbose=True )      # A mettre lorsque la description de la chaine est finalisée


# Génération d'une séquence de longueur Nsamples
obsSeq,statesSeq = model.sample(length=Nsamples,path=True)

obsSeq=np.array([dicoObs[t] for t in obsSeq])      # Conversion des observables String -> int (0:beau, 1 pluie) )
statesSeq=np.array([dicoState[t.name] for t in statesSeq[1:]])   # Conversion des etats String -> int (0:beau, 1 pluie) )

# Chargement des données du fichier RR5MN
#ObsMesure = mio.loadmat('RR5MN.mat')['Support'].astype(np.int8).squeeze()

