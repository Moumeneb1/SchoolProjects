
# maintenance
import warnings as wrn

# outils mathematiques de base
import math
import numpy as np

# affichage
import matplotlib
import matplotlib.pyplot as plt

# generation aleatoire
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import binom

def schioler(x):
    ''' Calcul de la fonction schioler définie comme suit :
          f(x) =  sin (pi.x)    sur  ] -1, 1 [
          f(x) =  0             sur  [ -2,-1 ]   U  [1,  2]
    '''
    return np.sin(np.pi*x)*np.logical_and(x>-1,x<1).astype(float)
def bruit_gaussien(size=1,mu=0,sigma=1):
    ''' bruit qui suit une distribution normale N(mu ; sigma**2)'''
    return norm.rvs(mu,sigma,size)
def schioler_bruite(x,sigma=0.2):
    return schioler(x) + bruit_gaussien(x.shape,mu=0,sigma=sigma)
def jeuDeDonnees_schioler_bruite(size,sigma=0.2) :
    ''' Calcul de la fonction schioler définie comme suit :
    | Sur les abscisses les points suivent une distribution uniforme sur [-2, 2];
    | y = f(x) + delta  où :
    |     f(x) =  sin (pi.x)    sur  ] -1, 1 [
    |     f(x) =  0             sur  [ -2,-1 ]   U  [1,  2]
    |     avec delta un bruit qui suit une distribution normale N(0 ; sigma**2)
    |
    |   X, Y = schioler(size,sigma)
    |
    | N     : Nombre de données (points d'abscisses) à générer
    | sigma : Sigma de la loi Normale (N(0 ; sigma**2)) suivie par le bruit à ajouter
    |         aux données (0.2 est la valeur par defaut qui correspond à l'énoncé)
    | En sortie :
    | X : Les valeurs d'abscisse tirées aléatoirement
    | Y : Les valeurs de sortie de la fonction
    '''       
    X = uniform.rvs(0,1,size)*4-2
    X = np.sort(X)
    return X, schioler_bruite(X,sigma=sigma)
def jeuDeDonnees_schioler_bruite_inhomogene(size,a1=-.75,b1=-.25,a2=1,b2=1.25,a3=None,b3=None,sigma=0.2) :
    ''' Calcul de la fonction schioler définie comme suit :
    | Sur les abscisses les points suivent une distribution uniforme ihomogene  
    | ainsi X est tire sur [a1, b1] U [a2, b2];
    | y = f(x) + delta  où :
    |     f(x) =  sin (pi.x)    sur  ] -1, 1 [
    |     f(x) =  0             sur  [ -2,-1 ]   U  [1,  2]
    |     avec delta un bruit qui suit une distribution normale N(0 ; sigma**2)
    |
    |   X, Y = schioler(size,sigma)
    |
    | N     : Nombre de données (points d'abscisses) à générer
    | sigma : Sigma de la loi Normale (N(0 ; sigma**2)) suivie par le bruit à ajouter
    |         aux données (0.2 est la valeur par defaut qui correspond à l'énoncé)
    | En sortie :
    | X : Les valeurs d'abscisse tirées aléatoirement
    | Y : Les valeurs de sortie de la fonction
    '''
    if a3 == None and b3 == None:
        if not (a1 < b1 < a2 < b2 ):
            raise ValueError("a1 < b1 < a2 < b2 pas verifie.")
        p1 = (b1-a1)/((b1-a1)+(b2-a2))
        size1 = binom.rvs(size,p1)
        size2 = size-size1
        X = np.concatenate((uniform.rvs(0,1,size1)*(b1-a1)+a1,uniform.rvs(0,1,size2)*(b2-a2)+a2))
    else:
        if not (a1 < b1 < a2 < b2 < a3 < b3 ):
            raise ValueError("a1 < b1 < a2 < b2 < a3 < b3 pas verifie.")
        p1 = (b1-a1)/((b1-a1)+(b2-a2)+(b3-a3))
        p2 = (b2-a2)/((b1-a1)+(b2-a2)+(b3-a3))
        size1 = binom.rvs(size,p1)
        size2 = binom.rvs(size,p2)
        size3 = size-size2-size1
        X = np.concatenate((uniform.rvs(0,1,size1)*(b1-a1)+a1,uniform.rvs(0,1,size2)*(b2-a2)+a2,uniform.rvs(0,1,size3)*(b3-a3)+a3))
    X = np.sort(X)
    return X, schioler_bruite(X,sigma=sigma)

def silverman(x):
    ''' Calcul de la fonction silverman définie comme suit :
    |     f(x) =  sin(2.pi(1-.x)**2)   sur  ] 0, 1 [
    |     f(x) =  0                    sur  [-0.25, 0 ]
    '''
    return np.sin(2*np.pi*(1-x)**2)*np.logical_and(x>0,x<1).astype(float)
def bruit_gaussien_nonstationnaire(x,mu=0,sigma=1):
    ''' bruit qui suit une distribution normale N(0 ; sigma(x)**2), avec
         avec la variance sigma(x)**2   =  0.0025   si x <= 0.05
                                        =  x**2     si x >  0.05'''
    if not isinstance(x,(int,float,np.ndarray)):
        raise NotImplementedError("Voir si cela est gerable")
    sigma = sigma * (x*(x>.05) + .05*(x<=.05))
    return norm.rvs(mu,sigma**2)
def silverman_bruite(x,sigma=0.2):
    return silverman(x) + bruit_gaussien_nonstationnaire(x,mu=0,sigma=sigma)
def jeuDeDonnees_silverman_bruite(size,sigma=0.05) :
    ''' Calcul de la fonction silverman définie comme suit :
    | Sur les abscisses les points suivent une distribution uniforme sur [-0.25, 1];
    | y = f(x) + delta(x)  où :
    |     f(x) =  sin(2.pi(1-.x)**2)   sur  ] 0, 1 [
    |     f(x) =  0                    sur  [-0.25, 0 ]
    |     delta(x) un bruit qui suit une distribution normale N(0 ; sigma(x)**2), avec
    |     avec la variance sigma(x)**2 =  0.0025   si x <= 0.05
    |                                  =  x**2     si x >  0.05
    | X, Y = silverman(N,sigma)
    | = 
    | N     : Nombre de données (points d'abscisses) à générer
    | sigma : Sigma de la loi Normale (N(0 ; sigma**2)) suivie par le bruit à ajouter
    |         aux données. Par défaut l'écart type (sigma) de 0.05 correspond à une
    |         variance de 0.0025
    | En sortie :
    | X : Les valeurs d'abscisse tirées aléatoirement
    | Y : Les valeurs de sortie de la fonction
    '''       
    X = uniform.rvs(0,1,size)*1.25-0.25;   
    X = np.sort(X)
    return X, silverman_bruite(X,sigma=sigma)

def affichage_illustratif_schioler():

    nombreDeDonnees = 1000
    xi,yi = jeuDeDonnees_schioler_bruite(nombreDeDonnees,sigma=.2)

    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-2,2,n_maillage).reshape(n_maillage,1) 

    plt.subplot(1,2,1)
    plt.plot(x_maillage,schioler(x_maillage),'b-',label='courbe théorique')
    plt.plot(xi,yi,'r.',alpha=.5,label='données bruitées')
    plt.axis([-2, 2, -2.75, 2.75])
    plt.xlabel('x []')
    plt.ylabel('y []')
    plt.legend()#['Ens App','Ens Val','f. theorique']);
    plt.grid(True)
    plt.title('Données "Schioler"')
    plt.subplot(1,2,2)
    plt.plot(x_maillage,bruit_gaussien(len(x_maillage),sigma=.2),'r.',label='bruit')
    plt.plot(x_maillage,schioler(x_maillage),'b-',alpha=.5,label='courbe théorique')
    plt.axis([-2, 2, -2.75, 2.75])
    plt.title('Bruit "Schioler"') ;
    plt.xlabel('x []')
    plt.ylabel('y []')
    plt.legend()#['Ens App','Ens Val','f. theorique']);
    plt.grid(True)
    plt.tight_layout()

def affichage_illustratif_silverman():
    
    nombreDeDonnees = 1000
    xi,yi = jeuDeDonnees_silverman_bruite(nombreDeDonnees,sigma=1)

    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-.5,1.25,n_maillage).reshape(n_maillage,1) 

    plt.subplot(1,2,1)
    plt.plot(x_maillage,silverman(x_maillage),'b-',label='courbe théorique')
    plt.plot(xi,yi,'r.',alpha=.5,label='données bruitées')
    plt.axis([-.5, 1.25, -2.75, 2.75])
    plt.xlabel('x []')
    plt.ylabel('y []')
    plt.legend()#['Ens App','Ens Val','f. theorique']);
    plt.grid(True)
    plt.title('Données "Silverman"')
    plt.subplot(1,2,2)
    plt.plot(x_maillage,bruit_gaussien_nonstationnaire(x_maillage,sigma=1),'r.',label='bruit')
    plt.plot(x_maillage,silverman(x_maillage),'b-',alpha=.5,label='courbe théorique')
    plt.axis([-.5, 1.25, -2.75, 2.75])
    plt.title('Bruit "Silverman"') ;
    plt.xlabel('x []')
    plt.ylabel('y []')
    plt.legend()#['Ens App','Ens Val','f. theorique']);
    plt.grid(True)
    plt.tight_layout()


def decoupageEnDeuxSousEnsembles(xi,yi,ratio,fonction=silverman):

    if not 0 < ratio < 1:
        raise ValueError("ratio doit etre un reel dans ]0,1[.")

    if not ( isinstance(xi,(np.ndarray)) and  isinstance(xi,(np.ndarray)) ):
        raise TypeError("xi ({}) et/ou yi ({}) n'ont pas le(s) bon(s) type(s).".format(type(xi),type(yi)))
    
    if len(xi) == len(yi):
        nombreDeDonnees = len(xi)
        
    index = np.random.permutation(nombreDeDonnees)
    nombreDeDonnees_app = math.floor(ratio*nombreDeDonnees)
    
    index_app = index[:nombreDeDonnees_app]
    index_app.sort()
    index_val  = index[nombreDeDonnees_app:]
    index_val.sort()

    x_app = xi[index_app] # ensemble d'Apprentissage
    y_app = yi[index_app] 

    x_val = xi[index_val] # ensemble de Validation
    y_val = yi[index_val] 

    return x_app, y_app, x_val, y_val

def affichage_ensembles(x_app, y_app, x_val, y_val, x_test, y_test,fonction):

    mini = math.floor(min([min(x_app),min(x_app),min(x_test)])*2)/2
    maxi = math.ceil(max([max(x_app),max(x_app),max(x_test)])*2)/2
    #print([mini,maxi],fonction.__name__)
    
    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(mini,maxi,n_maillage).reshape(n_maillage,1) 
    plt.plot(x_app, y_app,'b.',label='app.')
    plt.plot(x_val, y_val,'r.',label='val.')
    plt.plot(x_test, y_test,'m.',label='test.')
    plt.plot(x_maillage,fonction(x_maillage),'k-',alpha=.5,label='courbe théorique')
    #
    plt.xlabel('x []')
    plt.ylabel('y []')
    plt.legend()#['Ens App','Ens Val','f. theorique']);
    plt.grid(True)
    plt.tight_layout()


if __name__ == "__main__":

    #
    # quelques affichages pour un jeu de donnees simulees "schioler"
    # ---------------------------------------------------------------
    plt.figure(figsize=(12,6))
    affichage_illustratif_schioler()
    plt.tight_layout()
    
    # generation des jeux de données simulées
    # ensembles d'apprentissage et de validation
    nombreDeDonnees_appVal = 1000
    nombreDeDonnees_test = 500
    xi,yi = jeuDeDonnees_schioler_bruite(nombreDeDonnees_appVal,sigma=.2)
    r_l=.5 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val = decoupageEnDeuxSousEnsembles(xi,yi,r_l)

    # ensemble de test (performances en generalisation une fois l'apprentissage termine)
    x_test,y_test = jeuDeDonnees_schioler_bruite(nombreDeDonnees_test,sigma=.2)
    n_test = np.size(x_test,0)
    
    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-2,2,n_maillage).reshape(n_maillage,1) 

    # generation des ensembles d'apprentissage et de validation (performances en generalisation)
    r_l=.5 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val  = decoupageEnDeuxSousEnsembles(xi,yi,r_l,)
    n_app = np.size(x_app,0)
    n_val = np.size(x_val,0)

    
    plt.figure(figsize=(12,6))
    affichage_ensembles(x_app, y_app, x_val, y_val, x_test, y_test,fonction=schioler)
    plt.title('''Ensembles d'apprentissage ("schioler")''')
    plt.tight_layout()


    #
    # quelques affichages pour un jeu de donnees simulees "schioler"  [-2,-.75] U [-.25,1]  U [1.25,2] 
    # ---------------------------------------------------------------
    
    # generation des jeux de données simulées
    # ensembles d'apprentissage et de validation
    nombreDeDonnees_appVal = 1000
    nombreDeDonnees_test = 500
    a1=-2; b1=-.75 ; a2=-.25 ; b2=1 ; a3=1.25 ; b3=2
    xi,yi = jeuDeDonnees_schioler_bruite_inhomogene(nombreDeDonnees_appVal,a1=a1,b1=b1,a2=a2,b2=b2,a3=a3,b3=b3,sigma=.2)
    r_l=.5 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val = decoupageEnDeuxSousEnsembles(xi,yi,r_l)

    # ensemble de test (performances en generalisation une fois l'apprentissage termine)
    x_test,y_test = jeuDeDonnees_schioler_bruite(nombreDeDonnees_test,sigma=.2)
    n_test = np.size(x_test,0)
    
    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-2,2,n_maillage).reshape(n_maillage,1) 

    # generation des ensembles d'apprentissage et de validation (performances en generalisation)
    r_l=.5 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val  = decoupageEnDeuxSousEnsembles(xi,yi,r_l,)
    n_app = np.size(x_app,0)
    n_val = np.size(x_val,0)

    plt.figure(figsize=(12,6))
    affichage_ensembles(x_app, y_app, x_val, y_val, x_test, y_test,fonction=schioler)
    if a3 == None and b3 == None:
        titre = '''Ensembles d'apprentissage ("schioler" [{},{}] U [{},{}])'''.format(a1,b1,a2,b2)
    else:
        titre = '''Ensembles d'apprentissage ("schioler" [{},{}] U [{},{}] U [{},{}])'''.format(a1,b1,a2,b2,a3,b3)
    plt.title(titre)
    plt.tight_layout()



    #
    # quelques affichages pour un jeu de donnees simulees "schioler"  [-2,-.75,] U [-.25,1] 
    # ---------------------------------------------------------------
    
    # generation des jeux de données simulées
    # ensembles d'apprentissage et de validation
    nombreDeDonnees_appVal = 1000
    nombreDeDonnees_test = 500
    a1=-2 ; b1=-.75 ; a2=-.25 ; b2=1; a3=None; b3=None
    xi,yi = jeuDeDonnees_schioler_bruite_inhomogene(nombreDeDonnees_appVal,a1=a1,b1=b1,a2=a2,b2=b2,sigma=.2)
    r_l=.5 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val = decoupageEnDeuxSousEnsembles(xi,yi,r_l)

    # ensemble de test (performances en generalisation une fois l'apprentissage termine)
    x_test,y_test = jeuDeDonnees_schioler_bruite(nombreDeDonnees_test,sigma=.2)
    n_test = np.size(x_test,0)
    
    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-2,2,n_maillage).reshape(n_maillage,1) 

    # generation des ensembles d'apprentissage et de validation (performances en generalisation)
    r_l=.5 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val  = decoupageEnDeuxSousEnsembles(xi,yi,r_l,)
    n_app = np.size(x_app,0)
    n_val = np.size(x_val,0)

    
    plt.figure(figsize=(12,6))
    affichage_ensembles(x_app, y_app, x_val, y_val, x_test, y_test,fonction=schioler)
    if a3 == None and b3 == None:
        titre = '''Ensembles d'apprentissage ("schioler" [{},{}] U [{},{}])'''.format(a1,b1,a2,b2)
    else:
        titre = '''Ensembles d'apprentissage ("schioler" [{},{}] U [{},{}] U [{},{}])'''.format(a1,b1,a2,b2,a3,b3)
    plt.title(titre)
    plt.tight_layout()

    #
    # quelques affichages pour un jeu de donnees simulees "silverman"
    # ---------------------------------------------------------------
    plt.figure(figsize=(12,6))
    affichage_illustratif_silverman()
    plt.tight_layout()
    
    # generation des jeux de données simulées
    # ensembles d'apprentissage et de validation
    nombreDeDonnees_appVal = 1000
    nombreDeDonnees_test = 500
    xi,yi = jeuDeDonnees_silverman_bruite(nombreDeDonnees_appVal,sigma=1)
    r_l=.5 # proportions des differents ensembles de l'apprentissage 
    x_app, y_app, x_val, y_val = decoupageEnDeuxSousEnsembles(xi,yi,r_l)

    # ensemble de test (performances en generalisation une fois l'apprentissage termine)
    x_test,y_test = jeuDeDonnees_silverman_bruite(nombreDeDonnees_test,sigma=1)

    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-.5,1.25,n_maillage).reshape(n_maillage,1) 


    plt.figure(figsize=(12,6))
    affichage_ensembles(x_app, y_app, x_val, y_val, x_test, y_test,fonction=silverman)
    plt.title('''Ensembles d'apprentissage ("silverman")''')
    plt.tight_layout()
    
    plt.show()

