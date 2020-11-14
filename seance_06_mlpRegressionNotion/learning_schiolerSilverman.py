

#import time
#import math

# pour ignorer les alertes de la fonction apprentissage 
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from bruit_schiolerSilverman import schioler
from bruit_schiolerSilverman import jeuDeDonnees_schioler_bruite
from bruit_schiolerSilverman import jeuDeDonnees_schioler_bruite_inhomogene
from bruit_schiolerSilverman import affichage_illustratif_schioler

from bruit_schiolerSilverman import decoupageEnDeuxSousEnsembles
from bruit_schiolerSilverman import affichage_ensembles

from sklearn.neural_network import MLPRegressor

#from sklearn.base import clone
#from sklearn.model_selection import ShuffleSplit
#from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error

    
@ignore_warnings(category=ConvergenceWarning)
def apprentissage(rgrsr,x_app, y_app,x_val, y_val,max_n_iter=500):

    #    max_n_iter = 50000
    max_n_plateau = 500
    min_erreur_val = 10**-10
    
    listeDesMlp = list()
    listeDesScores_app = list()
    listeDesErreurs_app = list()
    listeDesScores_val = list()
    listeDesErreurs_val = list()

    err_val_min = float('inf')
    cpt_erreur_val_min = 0
 
    n_plateau = 0
    n_iter = 0
    cpt = 0
    while n_iter < max_n_iter or n_plateau < max_n_plateau : #and err_val_min > min_erreur_val: 
        rgrsr = rgrsr.fit(x_app, y_app.ravel())
        
        listeDesMlp.append(rgrsr)
        #listeDesMlp.append(clone(rgrsr))
        listeDesScores_app.append(rgrsr.score(x_app, y_app.ravel()))
        listeDesErreurs_app.append(mean_squared_error(y_app.ravel(),rgrsr.predict(x_app)))
        listeDesScores_val.append(rgrsr.score(x_val, y_val.ravel()))
        listeDesErreurs_val.append(mean_squared_error(y_val.ravel(),rgrsr.predict(x_val)))

        if cpt == 0 or listeDesErreurs_val[-1] < listeDesErreurs_val[cpt_erreur_val_min]:
        #if listeDesScores_val[-1] > listeDesScores_val[-2]:
            err_val_min = listeDesErreurs_val[-1]
            cpt_erreur_val_min = cpt
            n_plateau = 0
        else:
            n_plateau += 1
        n_iter += rgrsr.max_iter
        cpt += 1
        
    return listeDesMlp[cpt_erreur_val_min], listeDesErreurs_app, listeDesErreurs_val, cpt_erreur_val_min

def affichage_performances_apprentissage(listeDesErreurs_app,listeDesErreurs_val,cpt_erreur_val_min,rgrsr,xlabel=True,ylabel=True,legend=False):
    iterations_ = (1 + np.array(range(len(listeDesErreurs_val))) ) * rgrsr.max_iter
    iterations_min = (1 + cpt_erreur_val_min) * rgrsr.max_iter
    ax = plt.gca()
    ax.clear()
    ax.semilogy(iterations_,listeDesErreurs_app,':.r',label='app')
    ax.semilogy(iterations_,listeDesErreurs_val,':.g',label='val')
    y_lim = ax.get_ylim()
    ax.semilogy([iterations_min, iterations_min],y_lim,'k',linewidth=3,alpha=.5)
    ax.grid(True)
    ax.set_ylim(y_lim)
    if xlabel == True:
        ax.set_xlabel("nombre d'iterations/epochs")
    if ylabel == True:
        ax.set_ylabel("RMS")
    if legend == True:
        ax.legend()
    
def affichage_reseau_et_donnees(rgrsr,x_maillage,x_app,y_app,x_val,y_val,fonction,x_test=None,y_test=None,xlabel=True,ylabel=True,legend=True):
    ax = plt.gca()
    ax.plot(x_maillage,fonction(x_maillage),'k-',label='courbe théorique')
    ax.plot(x_app,y_app,'r.',alpha=.5,label='app')
    ax.plot(x_val,y_val,'g.',alpha=.5,label='val')
    if isinstance(x_test,(np.ndarray)) and  isinstance(y_test,(np.ndarray)):
        ax.plot(x_test,y_test,'m.',alpha=.5,label='test')
    elif x_test !=None or y_test != None:
            raise ValueError("argument non attendu")
        
    # courbe estimee
    ax.plot(x_maillage,rgrsr.predict(x_maillage),'b-',label='courbe estimee')
    ax.grid(True)
    if xlabel == True:
        ax.set_xlabel("x []")
    if ylabel == True:
        ax.set_ylabel("y []")
    if legend == True:
        ax.legend()
    
    return ax


def affichage_performances_et_donnees(erreurs_app,erreurs_val,cpt_erreur_val_min,n_hidden,rgrsr,x_app,y_app,x_val,y_val,fonction,x_maillage,
                                      x_test=None,y_test=None):

    ax = list()    
    if isinstance(x_test,(np.ndarray)) and  isinstance(y_test,(np.ndarray)) and isinstance(rgrsr,(list,MLPRegressor)):
        ncol = 5
    elif x_test ==None and y_test ==None:
        ncol = 4
    else:
        raise ValueError("argument non attendu")

    if not isinstance(n_hidden,list) and not isinstance(rgrsr,list):
        n_hidden = [n_hidden]
        erreurs_app = [erreurs_app]
        erreurs_val = [erreurs_val]
        cpt_erreur_val_min = [cpt_erreur_val_min]
        rgrsr = [rgrsr]
    elif isinstance(n_hidden,list) and isinstance(rgrsr,list):
        pass
    else:
        raise TypeError("Types incoherents pour n_hidden et rgrsr")
        
    for i,n in enumerate(n_hidden):
        ax.append(plt.subplot(len(n_hidden),2,1+2*i))
        if i != len(n_hidden)-1:
            affichage_performances_apprentissage(erreurs_app[i],
                                                 erreurs_val[i],
                                                 cpt_erreur_val_min[i],
                                                 rgrsr[i],
                                                 legend=False,xlabel=False)
        else:
            affichage_performances_apprentissage(erreurs_app[i],
                                                 erreurs_val[i],
                                                 cpt_erreur_val_min[i],
                                                 rgrsr[i],
                                                 legend=False,xlabel=True)


        ax.append(plt.subplot(len(n_hidden),2,2*(i+1)))
        #ax[-1].clear()
        if i != len(n_hidden)-1:
            ax[-1] = affichage_reseau_et_donnees(rgrsr[i],
                                                 x_maillage,
                                                 x_app,y_app,
                                                 x_val,y_val,fonction,
                                                 x_test=x_test,y_test=y_test,
                                                 legend=False,xlabel=False)
        else:
            ax[-1] = affichage_reseau_et_donnees(rgrsr[i],
                                                 x_maillage,
                                                 x_app,y_app,
                                                 x_val,y_val,fonction,
                                                 x_test=x_test,y_test=y_test,
                                                 legend=False,xlabel=True)
        
    fig = plt.gcf()
    bb = (fig.subplotpars.left, fig.subplotpars.top+0.02,fig.subplotpars.right-fig.subplotpars.left,.1)
    plt.legend(bbox_to_anchor=bb, mode="expand", loc="lower left",
                 ncol=5, borderaxespad=0., bbox_transform=fig.transFigure)
    #plt.tight_layout()


def affiche_rms(n_hidden,erreurs_app,erreurs_val,cpt_erreur_val_min,x_test=None,y_test=None,rgrsr=None):
    print('       m  err_app  err_val',end='')
    if isinstance(x_test,(np.ndarray)) and  isinstance(y_test,(np.ndarray)) and isinstance(rgrsr,(list,MLPRegressor)):
        print('      err_test',end='')
    else:
        if x_test !=None or y_test !=None or rgrsr !=None:
            raise ValueError("argument non attendu")
    print()
    
    if isinstance(n_hidden,(int)):
        print("{:8}  {:7.5f}  {:7.5f}".format(
            n_hidden,
            erreurs_app[cpt_erreur_val_min],
            erreurs_val[cpt_erreur_val_min]),end='')
        if isinstance(x_test,(np.ndarray)) and  isinstance(y_test,(np.ndarray)) and isinstance(rgrsr,(MLPRegressor)):
            print('     ({:7.5f})'.format(
                mean_squared_error(y_test.ravel(),rgrsr.predict(x_test))
            ),end='')
        print()
    elif isinstance(n_hidden,(list)):
        if cpt_erreur_val_min==None or not isinstance(cpt_erreur_val_min,(list)):
            raise ValueError(' liste "cpt_erreur_val_min" argument manquant.')
        for i,n in enumerate(n_hidden): 
            print("{:8}  {:7.5f}  {:7.5f}".format(
                n,
                erreurs_app[i][cpt_erreur_val_min[i]],
                erreurs_val[i][cpt_erreur_val_min[i]]),end="")
            if isinstance(x_test,(np.ndarray)) and  isinstance(y_test,(np.ndarray)) and isinstance(rgrsr,(list)):
                  print('     ({:7.5f})'.format(
                mean_squared_error(y_test.ravel(),rgrsr[i].predict(x_test))
                  ),end='')
            print()


if __name__ == '__main__':

    if False:
        affichage_illustratif_silverman()
    

    # generation des jeux de données simulées
    # ---------------------------------------
    
    # ensembles d'apprentissage et de validation
    nombreDeDonnees_appVal = 100
    nombreDeDonnees_test = 500
    xi,yi = jeuDeDonnees_schioler_bruite(nombreDeDonnees_appVal,sigma=.2)
    r_l=.1 # proportion de l'ensemble d'apprentissage 
    x_app, y_app, x_val, y_val = decoupageEnDeuxSousEnsembles(xi,yi,r_l)
    n_app = np.size(x_app,0)
    n_val = np.size(x_val,0)

    # ensemble de test (performances en generalisation une fois l'apprentissage termine)
    x_test,y_test = jeuDeDonnees_schioler_bruite(nombreDeDonnees_test,sigma=.2)
    n_test = np.size(x_test,0)

    # mise en forme pour MLPRegressor 
    x_app = np.reshape(x_app, [n_app, 1])
    x_val = np.reshape(x_val, [n_val, 1])
    x_test = np.reshape(x_test, [n_test, 1])

    
    # Abscisse régulière pour l'affichage des courbes
    n_maillage = 1000
    x_maillage = np.linspace(-2,2,n_maillage).reshape(n_maillage,1) 
    
    if False:
        affichage_ensembles(x_app, y_app, x_val, y_val, x_test, y_test,schioler)



    
    # apprentissage d'un mlp
    # ----------------------

    # Apprentissage pour un nombre de neurones en couche cachée
    n_hidden = 9
    # specification du modele
    rgrsr = MLPRegressor(activation='tanh', hidden_layer_sizes = (n_hidden), warm_start='True',solver='lbfgs', learning_rate = 'adaptive',max_iter=1)

    # apprentissage du modele
    rgrsr, erreurs_app, erreurs_val, cpt_erreur_val_min = apprentissage(rgrsr,x_app, y_app,x_val, y_val,max_n_iter=500)

    # affichage de l'apprentissage 
    fig= plt.figure(figsize=(12,6))
    affichage_performances_et_donnees(erreurs_app,erreurs_val,cpt_erreur_val_min,
                                      n_hidden,rgrsr,
                                      x_app,y_app,x_val,y_val,
                                      schioler,x_maillage)

    fig= plt.figure(figsize=(12,6))
    affichage_performances_et_donnees(erreurs_app,erreurs_val,cpt_erreur_val_min,
                                      n_hidden,rgrsr,
                                      x_app,y_app,x_val,y_val,
                                      schioler,x_maillage,
                                      x_test, y_test)

    affiche_rms(n_hidden,erreurs_app,erreurs_val,cpt_erreur_val_min)
    print()
    affiche_rms(n_hidden,erreurs_app,erreurs_val,cpt_erreur_val_min,x_test=x_test,y_test=y_test,rgrsr=rgrsr)
    print()
        


    # apprentissage de plusieurs mlp
    # ------------------------------

    # Apprentissage pour un nombre de neurones en couche cachée
    liste_n_hidden = list(range(1,10,2))

    # initialisation
    liste_rgrsr         = list()
    liste_erreurs_app = list()
    liste_erreurs_val = list()
    liste_cpt_erreur_val_min  = list()
        
    # specification du modele
    for i,n_hidden in enumerate(liste_n_hidden): 
        rgrsr = MLPRegressor(activation='tanh', hidden_layer_sizes = (n_hidden), warm_start='True',solver='lbfgs', learning_rate = 'adaptive',max_iter=1)

        # apprentissage du modele
        rgrsr, erreurs_app, erreurs_val, cpt_erreur_val_min = apprentissage(rgrsr,x_app, y_app,x_val, y_val,max_n_iter=500)
        liste_rgrsr.append(rgrsr)
        liste_erreurs_app.append(erreurs_app)
        liste_erreurs_val.append(erreurs_val)
        liste_cpt_erreur_val_min.append(cpt_erreur_val_min)

    # affichage de l'apprentissage
    fig= plt.figure(figsize=(12,6))
    affichage_performances_et_donnees(liste_erreurs_app,liste_erreurs_val,
                                      liste_cpt_erreur_val_min,
                                      liste_n_hidden,liste_rgrsr,
                                      x_app,y_app,x_val,y_val,
                                      schioler,x_maillage)
    
    fig= plt.figure(figsize=(12,6))
    affichage_performances_et_donnees(liste_erreurs_app,liste_erreurs_val,
                                      liste_cpt_erreur_val_min,
                                      liste_n_hidden,liste_rgrsr,
                                      x_app,y_app,x_val,y_val,
                                      schioler,x_maillage,
                                      x_test, y_test)


    affiche_rms(liste_n_hidden,liste_erreurs_app,liste_erreurs_val,liste_cpt_erreur_val_min)
    print()
    affiche_rms(liste_n_hidden,liste_erreurs_app,liste_erreurs_val,liste_cpt_erreur_val_min,x_test=x_test,y_test=y_test,rgrsr=liste_rgrsr)
    print()

    plt.show()

    
