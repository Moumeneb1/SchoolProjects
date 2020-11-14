#-----------------------------------------------------------
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm        # module palettes de couleurs
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
#-----------------------------------------------------------
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = (18, 6)
#np.set_printoptions(edgeitems=10)
np.set_printoptions(linewidth = 220)
np.set_printoptions(precision=4)
#np.set_printoptions(precision=3,formatter={'float': '{:9.3f}'.format})
pd.set_option('precision', 5)
pd.set_option("display.max_columns",20)
pd.set_option('display.max_rows', 999)
#pd.set_option('max_colwidth', 6)



linestyles = [(0, ()), # solid 
              (0, (5, 10)),(0, (5, 5)),(0, (5, 1)), # dashed (loosely/normal/densely)
              (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1)), # dotted  (loosely/normal/densely)
              (0, (3, 10, 1, 10, 1, 10)),(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1)), # dashdotted(loosely/normal/densely)
              (0, (1, 10)),(0, (1, 5)),(0, (1, 1))]
couleurs = cm.Dark2.colors


# chargement des donnees (avec pandas)
df = pd.read_csv('temperatures_enFrance.txt',sep=';',index_col=0)

# un numpy array des valeurs numeriques (pour la suite)
data = df._get_numeric_data().values
nomDesVilles = list(df.index)
nomDesVariables_ = list(df)
nomDesVariables_ = nomDesVariables_[:-1] # on ne considere que les variables quantitatives

# on met une partie des individus et des variables de cote
indice_individus = np.array(range(15))
indice_variables = np.array(range(12))

# broadcasting 
X = data[indice_individus[:,None],indice_variables] 

# List Comprehensions
nomDesVariables = [nomDesVariables_[i] for i in indice_variables]
nomDesIndividus = [nomDesVilles[i] for i in indice_individus]

#nombreDIndividus = np.size(X,axis=0)
#nombreDeVariables = np.size(X,axis=1)



indice_variables_sup = np.setdiff1d(np.array(range(0,len(nomDesVariables_))),indice_variables)         
nomDesVariables_sup = [nomDesVariables_[i] for i in indice_variables_sup] #nomDesVariables_[indice_variables_sup]
#X_var_sup = data[indice_individus,indice_variables_sup] 
X_var_sup = data[indice_individus[:,None],indice_variables_sup]
# comme on considère les corrélations il n'est pas necessaire de centrer ou de réduire les variables suppémentaires


# nettoyage des donnees chargees
# (on ne garde que les informations associees a l'acp.)
del data, nomDesVilles, nomDesVariables_
#%whos



moyennes = X.mean(axis=0)
moyennes



#ecartTypes = X.std(axis=0,ddof=0)
ecartTypes = X.std(axis=0,ddof=1)
ecartTypes



Xc = X - moyennes
Xcr = Xc / ecartTypes
#Xcr.std(axis=0,ddof=0)
#np.size(np.cov(Xcr.T,ddof=0),1)



donnees = pd.DataFrame(data=Xcr, index=nomDesIndividus, columns=nomDesVariables)
donnees.columns = [str(col) + '_cr' for col in donnees.columns]
#donnees


#la figure
for i, (ligne,label) in enumerate(zip(X, nomDesIndividus)):
    plt.plot(ligne, label=label,
             color = couleurs[i%len(couleurs)],
             linestyle=linestyles[(i//len(couleurs))%len(linestyles)])
plt.plot(moyennes,'k-',label="moyenne",linewidth=3)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True,fontsize=15)
plt.grid(True)
plt.xlabel("Mois de l'année")
plt.ylabel("Moyenne mensuelle de Température [$^oC$]") ;


#la figure
for i, (ligne,label) in enumerate(zip(Xc, nomDesIndividus)):
    plt.plot(ligne, label=label,
             color = couleurs[i%len(couleurs)],
             linestyle=linestyles[(i//len(couleurs))%len(linestyles)])
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True,fontsize=15)
plt.grid(True)
plt.xlabel("Mois de l'année")
plt.ylabel("Ecart à la Moyenne mensuelle de Température [$^oC$]") ;


#la figure
for i, (ligne,label) in enumerate(zip(Xcr, nomDesIndividus)):
    plt.plot(ligne, label=label,
             color = couleurs[i%len(couleurs)],
             linestyle=linestyles[(i//len(couleurs))%len(linestyles)])
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True,fontsize=15)
plt.grid(True)
plt.xlabel("Mois de l'année")
plt.ylabel("Ecart réduit à la Moyenne mensuelle de Température [$^oC$]") ;


ax = sns.heatmap(df.iloc[indice_individus,indice_variables].corr(), annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)



g = sns.pairplot(df.iloc[indice_individus,indice_variables], diag_kind="kde", markers="+",
                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                  diag_kws=dict(shade=True))



acp = PCA()
etude_centreeReduite = True
if etude_centreeReduite:
    CP = acp.fit(Xcr)
    lesNouvellesCoordonnees = acp.fit_transform(Xcr)
else:
    CP = acp.fit(X)
    lesNouvellesCoordonnees = acp.fit_transform(X)
# mise en forme
plesNouvellesCoordonnees = pd.DataFrame(data=lesNouvellesCoordonnees, index=nomDesIndividus, columns=list(range(1,acp.n_features_+1)))
plesNouvellesCoordonnees.columns = ['CP_' + str(col) for col in plesNouvellesCoordonnees.columns]
plesNouvellesCoordonnees

np.set_printoptions(precision=2)
print(" * valeurs propres des axes/facteurs (ordonnées) :\n",acp.singular_values_)
print(" * inerties des axes/facteurs (ordonnées) :\n",acp.singular_values_**2)
print(" * variance des axes/facteurs (ordonnées) :\n",acp.singular_values_**2/(np.size(X,axis=0)-1))
print(" * vecteurs propres :")
for i in range(len(acp.components_)):
    print("    v_{:<2d} :".format(i+1),acp.components_.T[:,i])
np.set_printoptions(precision=4)


#np.diag(np.cov(Xcr.T))
np.set_printoptions(precision=2)
print("variances :")
print(acp.explained_variance_," (facteurs)")
if etude_centreeReduite:
    print(np.diag(np.cov(Xcr.T,ddof=1))," (donnees)")
else:
    print(np.diag(np.cov(X.T,ddof=1))," (donnees)")

print("variance totale :")
print(" -",sum(acp.explained_variance_)," (facteurs)")
if etude_centreeReduite:
    print(" -",np.sum(np.diag(np.cov(Xcr.T,ddof=1)))," (donnees)")
else:
    print(" -",np.sum(np.diag(np.cov(X.T,ddof=1)))," (donnees)")
np.set_printoptions(precision=4)



np.set_printoptions(precision=2)
print("inerties :")
print(acp.explained_variance_*(acp.n_samples_-1),"(methode 1)")
print(acp.singular_values_**2,"(methode 2)")
#
print("inertie totale :")
print(np.sum(acp.explained_variance_*(acp.n_samples_-1)),"(methode 1)")
print(np.sum(acp.singular_values_**2),"(methode 2)")
np.set_printoptions(precision=4)
#
print("pourcentages d'inertie :\n",np.cumsum(acp.explained_variance_ratio_)*100)


plt.bar(np.arange(len(acp.explained_variance_ratio_))+1,acp.explained_variance_ratio_*100)
plt.plot(np.arange(len(acp.explained_variance_ratio_))+1,np.cumsum(acp.explained_variance_ratio_*100),'r--o')
plt.xlabel("Dimensions",fontsize=14)
plt.ylabel("% d'inertie expliquée",fontsize=14)
plt.title("Inertie expliquée en fonction du nombre de dimensions",fontsize=14);
plt.grid(True)


# nombres de dimensions
pourcentageDInertieSeuil = 95
d = np.argmax(np.cumsum(acp.explained_variance_ratio_)>=pourcentageDInertieSeuil/100)+1
print("Nombres de dimensions (>={:.0f}% inertie) : ".format(pourcentageDInertieSeuil),d)
#pourcentageDInertieSeuil = 99
#d = np.argmax(np.cumsum(acp.explained_variance_ratio_)>=pourcentageDInertieSeuil/100)+1
#print("Nombres de dimensions (>={:.0f}% inertie) : ".format(pourcentageDInertieSeuil),d)



qual = lesNouvellesCoordonnees*lesNouvellesCoordonnees
qual = (qual.T / qual.sum(axis=1)).T
qualite = pd.DataFrame(data=qual, index=nomDesIndividus, columns=list(range(1,acp.n_features_+1)))
del qual
#qualite.add_prefix('CP_')
qualite.columns = ['CP_' + str(col) for col in qualite.columns]
qualite*100



qualite.sum(axis=1)



contr = lesNouvellesCoordonnees*lesNouvellesCoordonnees
contr = contr / contr.sum(axis=0)
contribution = pd.DataFrame(data=contr, index=nomDesIndividus, columns=list(range(1,acp.n_features_+1)))
del contr
contribution.columns = ['CP_' + str(col) for col in contribution.columns]
contribution*100



contribution.sum(axis=0)


corrOldNew = np.corrcoef(X.T,lesNouvellesCoordonnees.T)
corrOldNew = corrOldNew[0:len(nomDesVariables),len(nomDesVariables):]
coordonneesDesVariables = pd.DataFrame(data=corrOldNew,
                                       index=nomDesVariables,
                                       columns=list(range(1,acp.n_features_+1)))
del corrOldNew
coordonneesDesVariables.columns = ['CP_' + str(col) for col in coordonneesDesVariables.columns]
coordonneesDesVariables


qualVar = coordonneesDesVariables**2
qualVar*100





qualVar.sum(axis=1)





contrVar=(coordonneesDesVariables**2)/(coordonneesDesVariables**2).sum(axis=0)
contrVar*100





contrVar.sum(axis=0)






corrOldNew_sup = np.corrcoef(X_var_sup.T,lesNouvellesCoordonnees.T)
corrOldNew_sup = corrOldNew_sup[0:len(nomDesVariables_sup),len(nomDesVariables_sup):]
coordonneesDesVariables_sup = pd.DataFrame(data=corrOldNew_sup,
                                       index=nomDesVariables_sup,
                                       columns=list(range(1,acp.n_features_+1)))
del corrOldNew_sup
coordonneesDesVariables_sup.columns = ['CP_' + str(col) for col in coordonneesDesVariables_sup.columns]
coordonneesDesVariables_sup





qualVar_sup = coordonneesDesVariables_sup**2
qualVar_sup*100







qualVar_sup.sum(axis=1)







# coordonnees maximales de chacune des figures
x_lim = [-1.1,1.1]
y_lim = [-1.1,1.1]
cpt = 0
plt.subplots(figsize=(10,10*d))
for i in range(d-1):
    for j in range(i+1,d):
        cpt += 1
        ax = plt.subplot('{}{}{}'.format(int(d*(d-1)/2),1,cpt))
        # cercle unitaire
        cercle = plt.Circle((0,0),1,color='red',fill=False)
        ax.add_artist(cercle)
        # t = np.linspace(0, 2*math.pi, 100)
        # a = 1;b = 1;u = v = 0
        # plt.plot( u+a*np.cos(t) , v+b*np.sin(t) )
        #
        # projection du nuage des variables 
        for k in range(len(nomDesVariables)):
            #ax.arrow(0, 0, corrOldNew[k][i], corrOldNew[k][j],length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
            ax.arrow(0, 0, coordonneesDesVariables.iloc[k,i], coordonneesDesVariables.iloc[k,j],length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
            # Ornementation
            #plt.text(corrOldNew[k][i], corrOldNew[k][j], nomDesVariables[k])#,fontsize=fontsize)
            plt.text(coordonneesDesVariables.iloc[k,i], coordonneesDesVariables.iloc[k,j], nomDesVariables[k])#,fontsize=fontsize)
        if not coordonneesDesVariables_sup.shape[0] == 0:
            for k in range(len(nomDesVariables_sup)):
                ax.arrow(0, 0, coordonneesDesVariables_sup.iloc[k,i], coordonneesDesVariables_sup.iloc[k,j],length_includes_head=True, head_width=0.05, head_length=0.1, fc='b', ec='b')
                # Ornementation
                plt.text(coordonneesDesVariables_sup.iloc[k,i], coordonneesDesVariables_sup.iloc[k,j], nomDesVariables_sup[k])#,fontsize=fontsize)
        plt.title('Axes {} et {}'.format(i+1,j+1))
        #
        # ajout d'une grille
        plt.grid(color='lightgray',linestyle='--')
        # Ajouter des deux axes correspondants aux axes factoriels
        ax.arrow(x_lim[0], 0, x_lim[1]-x_lim[0], 0,length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.plot(plt.xlim(), np.zeros(2),'k-')
        plt.text(x_lim[1], 0, "axe {:d}".format(i+1))
        #
        ax.arrow(0, y_lim[0], 0, y_lim[1]-y_lim[0],length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.plot(np.zeros(2),plt.ylim(),'k-')
        plt.text(0,y_lim[1], "axe {:d}".format(j+1))
        #        ax.set_ylim([-1.1, 1.1])
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')





cpt = 0
plt.subplots(figsize=(18,6*d))
for i in range(d-1):
    for j in range(i+1,d):
        cpt += 1
        ax = plt.subplot('{}{}{}'.format(int(d*(d-1)/2),1,cpt))
        plt.plot(lesNouvellesCoordonnees[:,i],lesNouvellesCoordonnees[:,j],'o')
        plt.title('Axes {} et {}'.format(i+1,j+1))
        if len(nomDesIndividus) != 0 :
            for k in  range(len(nomDesIndividus)):
                plt.text(lesNouvellesCoordonnees[k,i], lesNouvellesCoordonnees[k,j], nomDesIndividus[k])#,fontsize=fontsize)
        # Ajouter les axes
        plt.grid(color='lightgray',linestyle='--')
        x_lim = plt.xlim()
        ax.arrow(x_lim[0], 0, x_lim[1]-x_lim[0], 0,length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.plot(plt.xlim(), np.zeros(2),'k-')
        plt.text(x_lim[1], 0, "axe {:d}".format(i+1))
        y_lim = plt.ylim()
        ax.arrow(0,y_lim[0], 0, y_lim[1]-y_lim[0],length_includes_head=True, head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.plot(np.zeros(2),plt.ylim(),'k-')
        plt.text(0,y_lim[1], "axe {:d}".format(j+1))



# on reprend l'acp en limitant le nombre de facteurs
acp_ = PCA(n_components=d)
acp_.fit(Xcr)
# on determine le Xcr reconsrtuit a partir d'un nombre reduit de composantes de l'acp
Xcr_ = acp_.fit_transform(Xcr).dot(acp_.components_)+acp_.mean_
# et on en deduit le X reconstruit
X_ = Xcr_*ecartTypes+moyennes
X_= pd.DataFrame(data=X_, index=nomDesIndividus, columns=nomDesVariables)
#help(acp.fit_transform)
X_.columns = ['~' + str(col) for col in X_.columns]
X_






df.iloc[indice_individus,indice_variables]





X_.columns = [col[1:] for col in X_.columns] ## on modifie l'intitule pour que les deux colonnes aient bien le meme intitule
erreurDeTroncature = df.iloc[indice_individus,indice_variables]-X_ ## erreur de reconstruction
X_.columns = ['~' + str(col) for col in X_.columns] ## on remet l'intitule originale
erreurDeTroncature



pd.set_option('precision', 2)
erreurDeTroncature.describe()


plt.show()
