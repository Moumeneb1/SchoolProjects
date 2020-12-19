# From Sompy package
# Version 1.1 modifiée LATMOS L. Barthes / Thomas Beratto 02/11/2020

from view import MatplotView
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

class DendrogramView(MatplotView):
    
    def show(self, som):
        
        self.prepare()

        if False: # ce qu'il y avait avant
            Z = linkage(np.triu(som._distance_matrix),'ward','euclidean');
            dendrogram(Z,som.codebook.nnodes,'lastp');
        else:
             selec_mask = np.where(som.mask==1)[0]  
             Z = linkage(som.codebook.matrix[:,selec_mask],'ward','euclidean')
             plt.axis('on')
             dendrogram(Z,som.codebook.nnodes,'lastp')
        plt.show()
        
        #return 1

