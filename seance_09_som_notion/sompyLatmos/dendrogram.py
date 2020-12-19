# From Sompy package
# Version 1.1 modifiée LATMOS L. Barthes / Thomas Beratto 02/11/2020

from view import MatplotView
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

class DendrogramView(MatplotView):
    
    def show(self, som):
        
        self.prepare()
        
        Z = linkage(np.triu(som._distance_matrix),'ward','euclidean');
        dendrogram(Z,som.codebook.nnodes,'lastp');
        
        plt.show()
        
        return 1

