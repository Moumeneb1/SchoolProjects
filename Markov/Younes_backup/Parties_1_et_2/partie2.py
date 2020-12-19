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
start_probability = np.array([0.5, 0.4, 0.1])
T = np.array([[0.9, 0.1, 0],[0.5, 0.4, 0.1],[0, 0.35, 0.65]])                # Matrice de transition temporaire

B = np.array([[1,0],[0.8,0.2],[0,1]])               # matrice d'émission temporaire


dicoObs={'sun':0,'rain':1}        # pour transformer les chaines en entier (0,1 et 2)
dicoState={'c.sky':0,'cloudy':1,'v.cloudy':2}

## Creation de la chaine de Markov
model = pg.HiddenMarkovModel( name="partie 2" )      # Creation instance
# Matrice d'emission

# Creation etat beau temps et prob emission
sunny = pg.State( pg.DiscreteDistribution({ 'sun': B[0,0],'rain': B[0,1]}), name='c.sky' )   


# Creation etat beau temps et prob emission
cloudy = pg.State( pg.DiscreteDistribution({ 'sun': B[1,0],'rain': B[1,1]}), name='cloudy' )   


# Creation etat beau temps et prob emission
v_cloudy = pg.State( pg.DiscreteDistribution({ 'sun': B[2,0],'rain': B[2,1]}), name='v.cloudy' )   


# Matrice de transition
model.add_transitions(model.start,[sunny,cloudy,v_cloudy],[0.5, 0.4, 0.1])  # Probs initiales 
model.add_transitions(sunny,    [sunny,cloudy,v_cloudy], [T[0,0],T[0,1],T[0,2]])     # transitions depuis sunny
model.add_transitions(cloudy,   [sunny,cloudy,v_cloudy], [T[1,0],T[1,1],T[1,2]])     # transition depuis rainy
model.add_transitions(v_cloudy, [sunny,cloudy,v_cloudy], [T[2,0],T[2,1],T[2,2]])     # transition depuis rainy

model.add_transition( model.start, sunny, start_probability[0] )       
model.add_transition( model.start, cloudy, start_probability[1] )   
model.add_transition( model.start, v_cloudy, start_probability[2] ) 

model.bake( verbose=True )      # A mettre lorsque la description de la chaine est finalisée


# Génération d'une séquence de longueur Nsamples
obsSeq,statesSeq = model.sample(length=Nsamples,path=True)

obsSeq=np.array([dicoObs[t] for t in obsSeq])      # Conversion des observables String -> int (0:beau, 1 pluie) )
statesSeq=np.array([dicoState[t.name] for t in statesSeq[1:]])   # Conversion des etats String -> int (0:beau, 1 pluie) )

# Chargement des données du fichier RR5MN
ObsMesure = mio.loadmat('RR5MN.mat')['Support'].astype(np.int8).squeeze()

