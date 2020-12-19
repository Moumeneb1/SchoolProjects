import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
#import matplotlib.gridspec as gridspec

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# routine d'affichage pour suivre l'apprentissage du reseau 
# a l'aide des ses poids
try: 
    from sklearn.manifold import TSNE
    HAS_SK = True
except: 
    HAS_SK = False
    print('Please install sklearn for layer visualization')


import torch

    
# fonction d'affichage des matrices de confusion (copié/collé du site de scikit-learn)
# ----------------------------------------------

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          fontsize=16):
    """
    This function printed and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Attention : les classes commencent à zero
    copier/coller d'un tutoriel sklearn?
    """
    if title is None:
        title=""
    if isinstance(title,str) and title == "":
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # suppose que les classes sont numerotees à partir de 0
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [ classes[i] for i in unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
    #    print('Confusion matrix, without normalization')

    #print(cm)

    #fig, ax = plt.subplots()
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0])
           # ... and label them with the respective list entries
           #xticklabels=classes, yticklabels=classes
           #title=title,
           #ylabel='True label',
           #xlabel='Predicted label'
          )
    ax.set_title(title,fontsize=fontsize)
    ax.set_xlabel('Predicted label',fontsize=fontsize)
    ax.set_xticklabels(classes,fontsize=fontsize)
    ax.set_ylabel('True label',fontsize=fontsize)
    ax.set_yticklabels(classes,fontsize=fontsize)
    
    ## Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor",fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",fontsize=fontsize,
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def get_accuracy(predictions, true_labels):
    if isinstance(predictions,torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(true_labels,torch.Tensor):
        true_labels = true_labels.numpy()
    if len(predictions.shape) != 1 :
        if predictions.shape[1] == 1 or  predictions.shape[0] == 1:
            predictions = np.squeeze(predictions)
    if len(true_labels.shape) != 1 :
        if true_labels.shape[1] == 1 or  true_labels.shape[0] == 1:
            true_labels = np.squeeze(true_labels)
    
    corrects = (predictions == true_labels).sum()
    accuracy = 100.0 * corrects/len(true_labels)
    return accuracy


    
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s.__str__(), backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')


def affichageDesPerformancesDuReseau(cnn, classes,
                                     train_data_, train_targets_,
                                     validation_data_,validation_targets_,
                                     test_data_,test_targets_):

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    
    #accuracy = float((validation_predictions_ == validation_targets_.data.numpy()).astype(int).sum()) / float(validation_targets_.size(0))
    #print( '| validation accuracy: %.2f' % accuracy)
     
    # Visualization of trained flatten layer (T-SNE)
    plot_only = 500
    plt.figure(figsize=(15,5))
    #
    train_output_, last_layer = cnn(train_data_)
    train_predictions_ = torch.max(train_output_, 1)[1].data.numpy()
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
    labels = train_targets_.numpy()[:plot_only]
    #
    plt.subplot(2,3,1)
    plot_with_labels(low_dim_embs, labels)
    plt.subplot(2,3,1+3)
    titre = "train ({:.1f}%)".format(get_accuracy(train_predictions_,train_targets_))
    plot_confusion_matrix(train_targets_, train_predictions_, classes,
                          title=titre,fontsize=12)
    #
    #
    validation_output_, last_layer = cnn(validation_data_)
    validation_predictions_ = torch.max(validation_output_, 1)[1].data.numpy()
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
    labels = validation_targets_.numpy()[:plot_only]
    #
    plt.subplot(2,3,2)
    plot_with_labels(low_dim_embs, labels)
    plt.subplot(2,3,2+3)
    titre = "validation ({:.1f}%)".format(get_accuracy(validation_predictions_,
                                                       validation_targets_.numpy()))
    plot_confusion_matrix(validation_targets_, validation_predictions_, classes,
                          title=titre,fontsize=12)
    #
    #
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    test_output_, last_layer = cnn(test_data_)
    test_predictions_ = torch.max(test_output_, 1)[1].data.numpy()
    low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
    labels = test_targets_.numpy()[:plot_only]
    #
    ax=plt.subplot(2,3,3)
    plot_with_labels(low_dim_embs, labels)
    plt.subplot(2,3,3+3)
    titre = "test ({:.1f}%)".format(get_accuracy(test_predictions_,test_targets_.numpy()))
    plot_confusion_matrix(test_targets_, test_predictions_, classes,
                          title=titre,fontsize=12)


   
