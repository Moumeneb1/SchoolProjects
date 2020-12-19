import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from scipy.spatial.transform import Rotation

def  splitNlabs(N,classnames=None) :
    N1 = int(np.ceil(N/3)); 
    N2 = int(np.floor(N/3)); 
    N3 = N - N1 - N2;
    labs=np.empty(N).astype(str)
    if classnames is not None : 
       labs[0:N1]     = classnames[0]; 
       labs[N1:N1+N2] = classnames[1]; 
       labs[N1+N2:N]  = classnames[2]; 
    else : # noms des classes par defaut
       labs[0:N1]     = ['A']; # %{'A'};  %{'un'};    %{'C1'}
       labs[N1:N1+N2] = ['B']; # %{'B'};  %{'deux'};  %{'C2'}
       labs[N1+N2:N]  = ['C']; # %{'C'};  %{'trois'}; %{'C3'}
    cnames = ['X', 'Y'];   
    return N1,N2,N3,labs,cnames

def Zcreadata(N,classnames=None) :
    ''' Creation d'un jeu de donnees simulees en dimension 2 et en
    % forme de lettre Z.
    €n Entree :
    % N          : Nombre de points de donnees
    % classnames : Vecteur des noms des classes
    % Sorties :
    % X      : Jeu de donnees 2D en forme de lettre Z
    % labs   : Labels des donnees selon 3 classes chacune
    %          associee aux points selon leurs appartenances
    %          aux barres qui forment la lettre Z :
    %            - barre du haut   :<->: 1ere classe
    %            - barre du bas    :<->: 2eme classe
    %            - barre diagonale :<->: 3eme classe
    % cnames : Noms des variables des 2 dimensions
    '''
    N1,N2,N3,labs,cnames = splitNlabs(N,classnames)
    X1 = np.random.uniform((-.5,.3),(.5,.5),(N1,2))
    X2 = np.random.uniform((-.5,-.3),(.5,-.5),(N2,2))
    X3 = np.random.uniform((-.5,-.1),(0.5,.1),(N3,2))
    a = 30
    R = np.array([ [np.cos(a/180*np.pi), np.sin(a/180*np.pi)],
                   [- np.sin(a/180*np.pi), np.cos(a/180*np.pi)] ])
    X3 = X3@R
    X  = np.concatenate((X1,X2,X3));
    
    return X, labs, cnames

def Fcreadata(N,classnames=None) :
    ''' Creation d'un jeu de donnees simulees en dimension 2 et en
    % forme de lettre f.
    €n Entree :
    % N          : Nombre de points de donnees
    % classnames : Vecteur des noms des classes
    % Sorties :
    % X      : Jeu de donnees 2D en forme de lettre f
    % labs   : Labels des donnees selon 3 classes chacune
    %          associee aux points selon leurs appartenances
    %          aux barres qui forment la lettre F :
    %            - barre du haut   :<->: 1ere classe
    %            - barre du milieu :<->: 2eme classe
    %            - barre de gauche :<->: 3eme classe
    % cnames : Noms des variables des 2 dimensions
    '''
    N1,N2,N3,labs,cnames = splitNlabs(N,classnames)
    X1 = np.random.uniform((.2,.8),(1,1),(N1,2))
    X2 = np.random.uniform((.2,.4),(1,.6),(N2,2))
    X3 = np.random.uniform((0,0),(0.2,1),(N3,2))
    #                        ax    ay   bx    by
    #X1 = tls.gen2duni(N1,[0.2,  1.0],[0.5, 1.5], 0);   
    #X2 = tls.gen2duni(N2,[0.2,  0.0],[0.5, 0.5], 0);   
    #X3 = tls.gen2duni(N3,[0.1, -1.0],[0.2, 1.5], 0);   
    X  = np.concatenate((X1,X2,X3));    
    return X, labs, cnames

def lettreplot(Data) :
    ''' Affichage des donnees qui representent la lettre Z ou F
    % en 3 classes selon un schema de partitionnement pre-etabli.
    % Entrees :
    % Data    : Donnees a afficher
    %
    % Note : Il n'y a pas de passage de parametre pour le choix des
    %        couleurs ou des formes. L'utilisateur qui le souhaite
    %        peut a son gre changer cela ou intervenir directement
    %        dans ce code.
    '''
    Ndata=len(Data);
    #
    N1 = int(np.ceil(Ndata/3));   # Garder la coherence de cette ...
    N2 = int(np.floor(Ndata/3));  # ... partition avec les fonctions ... 
    N3 = Ndata - N1 - N2;         # ... Zcreadata et Fcreadata.
    #
    Dx = Data[0:N1,:]
    plt.scatter(Dx[:,0],Dx[:,1],marker='*',s=50,c='c')
    Dx = Data[N1:N1+N2,:]
    plt.scatter(Dx[:,0],Dx[:,1],marker='^',s=50,c='g');
    Dx = Data[N1+N2:Ndata,:]; 
    plt.scatter(Dx[:,0],Dx[:,1],marker='o',s=50,c='b');
    plt.axis( [min(Data[:,0])-0.1, max(Data[:,0])+0.1, min(Data[:,1])-0.1, max(Data[:,1])+0.1]);
