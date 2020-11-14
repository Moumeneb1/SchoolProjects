#-----------------------------------------------------------
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm        # module palettes de couleurs
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


class EffetDeSerre:
    noms_des_mesures = ('t2','tcc','lsp','cp','ssr','CO2') 
    noms_des_lieux   = ('Reykjavik','Oslo','Paris','New York','Tunis','Alger','Beyrouth','Atlan','Dakar')
    noms_des_mois    = ('janv','fév','mars','avril','mai','juin','juil','aout','sept','oct','nov','déc')

    def __init__(self):
        # informations associees aux donnees
        
        # Chargement des données  (période 1/1982 à 10/2011)
        clim_t2  = np.array(scipy.io.loadmat('clim_t2C_J1982D2010')['clim_t2'])  # Temperature at 2 meters (degC) 
        clim_tcc = np.array(scipy.io.loadmat('clim_tcc_J1982D2010')['clim_tcc']) # Total cloud cover (0-1)
        clim_lsp = np.array(scipy.io.loadmat('clim_lsp_J1982D2010')['clim_lsp']) # Large scale precipitation (m) 
        clim_cp  = np.array(scipy.io.loadmat('clim_cp_J1982D2010')['clim_cp'])   # Convective precipitation (m) 
        clim_ssr = np.array(scipy.io.loadmat('clim_ssr_J1982D2010')['clim_ssr']) # Surface solar radiation ((W/m^2)s)
        clim_co2 = np.array(scipy.io.loadmat('clim_co2_J1982D2010')['clim_co2']) # CO2 ppm

        # on genere les indices temporels associées aux données
        assert(np.all(clim_t2[:,0] == clim_tcc[:,0]) and np.all(clim_t2[:,0] == clim_lsp[:,0])
               and np.all(clim_t2[:,0] == clim_cp[:,0]) and np.all(clim_t2[:,0] == clim_ssr[:,0])
               and np.all(clim_t2[:,0] == clim_co2[:,0]))
        annee = clim_t2[:,0].astype(int)
        assert(np.all(clim_t2[:,1] == clim_tcc[:,1]) and np.all(clim_t2[:,1] == clim_lsp[:,1])
               and np.all(clim_t2[:,1] == clim_cp[:,1]) and np.all(clim_t2[:,1] == clim_ssr[:,1])
               and np.all(clim_t2[:,1] == clim_co2[:,1]))
        mois = clim_t2[:,1].astype(int)

        # nettoyage des colonnes de temps
        clim_t2  = np.delete(clim_t2,  [0,1], axis=1)
        clim_tcc = np.delete(clim_tcc, [0,1], axis=1)
        clim_lsp = np.delete(clim_lsp, [0,1], axis=1)
        clim_cp  = np.delete(clim_cp,  [0,1], axis=1)
        clim_ssr = np.delete(clim_ssr, [0,1], axis=1)
        clim_co2 = np.delete(clim_co2, [0,1], axis=1)

        # on met les données en forme
        data = np.column_stack((clim_t2, clim_tcc, clim_lsp, clim_cp, clim_ssr))
        del clim_t2, clim_tcc, clim_lsp, clim_cp, clim_ssr

        # creation de la structure de données pandas
        mux = pd.MultiIndex.from_product((self.noms_des_mesures[:-1], self.noms_des_lieux))

        ## on cree la serie temporelle pandas correspondante
        if False: #True: # 
            # temps pour pandas
            time_index = pd.to_datetime({'year':annee,'month':mois,'day':mois*0+1})
            self.df = pd.DataFrame(data, columns=mux,index=time_index)
        else:
            self.df = pd.DataFrame(data, columns=mux,index=[annee,mois])
        self.df[self.noms_des_mesures[-1]] = clim_co2[:,0]
        del data, clim_co2

    def transpose(self):
        self.df = self.df.transpose()
        
    def seLimiterAUnLieu(self,nomDuLieu=None):
        ''' extraction des données pour une ville donnée '''

        '''Remarque: 
           * les colonnes de la sont des tuples (nom de ville, nom de mesure)'''

        if nomDuLieu==None:
            nomDuLieu = self.noms_des_lieux[0]

        # selection des colonnes correspondant a la ville (pas propre, il doit y avoir une fonction pandas) 
        selec_columns = np.where(self.df.columns.codes[1] == np.where(self.df.columns.levels[1]==nomDuLieu)[0])
        selec_columns = (np.array(np.append(selec_columns[0],-1)),)
        # [0] a la fin pour sortir d'un array a un element
        
        # on extrait la structure contenant ce qui nous interesse
        self.df = self.df.iloc[:,selec_columns[0]] # [0] pour sortir d'un array a un element
        self.df.columns = [ i[0] for i in list(self.df.columns)]


    def seLimiterAUneMesure(self,nomDeLaMesure=None):
        ''' extraction des données pour une mesure donnée '''

        '''Remarque: 
           * les colonnes de la sont des tuples (nom de ville, nom de mesure)'''

        if nomDeLaMesure==None:
            nomDeLaMesure = self.noms_des_mesures[0]
            
        # selection des colonnes correspondant a la mesure (pas propre, il doit y avoir une fonction pandas)
        selec_columns = np.where(self.df.columns.codes[0] == np.where(self.df.columns.levels[0]==nomDeLaMesure)[0])
        # [0] a la fin pour sortir d'un array a un element

        # on extrait la structure contenant ce qui nous interesse
        self.df = self.df.iloc[:,selec_columns[0]]
        self.df.columns = [ i[1] for i in list(self.df.columns)]


    def passerAUneClimatologie(self,nomDesMois=None):
        ''' Comportement moyenné sur le mois de l'année 
              -> 12 valeurs (une par mois de l'année)
        '''
        if nomDesMois==None:
            nomDesMois = EffetDeSerre.noms_des_mois
            
        #print('rr',self.df.index.get_level_values(1))
        self.df = self.df.groupby(self.df.index.get_level_values(1)).mean()
        self.df.index = (nomDesMois[i-1] for i in self.df.index)

        
    def passerADesMoyennesAnnuelles(self):
        ''' Comportement moyen pour chaque année 
              -> une valeur par année (moyenne sur les douze mois de l'année considérée)
        '''
        self.df = self.df.groupby(self.df.index.get_level_values(0)).mean()

if __name__ == "__main__":

    # divers parametres d'affichage
    # -----------------------------
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


    # différentes choses que l'on peut faire
    choix_climatologie      = True  # False # 
    choix_moyennesAnnuelles = False # True  # 

    choix_regarderUnLieu  = False
    choix_regarderUneMesure = False

    choix_regarderUnLieu  = True
    if choix_regarderUnLieu:
        # chargement des donnees
        lesDonnees = EffetDeSerre()
        if choix_climatologie:
            lesDonnees.passerAUneClimatologie()
        elif choix_moyennesAnnuelles:
            lesDonnees.passerADesMoyennesAnnuelles()
        lesNomsDeLieux = lesDonnees.noms_des_lieux
        indiceLieu = 0
        lesDonnees.seLimiterAUnLieu(lesNomsDeLieux[indiceLieu])

        plt.figure()
        ax = sns.heatmap(lesDonnees.df.corr(), annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
        plt.title(lesNomsDeLieux[indiceLieu])


    choix_regarderUneMesure = True
    if choix_regarderUneMesure:
        # chargement des donnees
        lesDonnees = EffetDeSerre()
        if choix_climatologie:
            lesDonnees.passerAUneClimatologie()
        elif choix_moyennesAnnuelles:
            lesDonnees.passerADesMoyennesAnnuelles()
        lesNomsDeMesures = lesDonnees.noms_des_mesures
        indiceMesure = 0
        lesDonnees.seLimiterAUneMesure(lesNomsDeMesures[indiceMesure])

        plt.figure()
        ax = sns.heatmap(lesDonnees.df.corr(), annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
        plt.title(lesNomsDeMesures[indiceMesure])

    plt.show()
