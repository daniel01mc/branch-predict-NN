import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.multiclass import unique_labels



def read_data(filename):

    df = pd.read_csv(filename,  delimiter=' ', header=None, names=['PC', 'Branch'],
                     converters={'PC': lambda x: int(x, 16),
                                 'Branch': lambda x: 1 if x == 't' else 0})
    df['Diff'] = df['PC'].diff()
    df.fillna(0, inplace=True)

    return df.to_dict(orient='list')

def evaluate(y_true, y_pred, name='', plot=False, normalize=False):
    """ Compute metrics between predicted and true labels """
    """ Compute Misprediction Rate """

    c_total = 0
    print('y pred', len(y_pred))
    print('y true', len(y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("True negative (TN):", tn)
    print("False positive (FP):", fp)
    print("False negative (FN):", fn)
    print("True positive (TP):", tp)

    total_mispredictions = fp + fn
    total_predictions = tp + tn

    print("total_mispredictions", total_mispredictions)
    print("total_predictions", total_predictions)

    c_total = (total_mispredictions  / total_predictions)*100
    print("Misprediction Rate: ", c_total)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    metrics = {
            'Accuracy': np.float16(acc),
            'Precision': np.float16(prec),
            'Recall':np.float16(rec),
            'F1-Score': np.float16(f1)
            }
    
    if plot:
        plot_confusion_matrix(y_true, y_pred, classes=['Not Taken', 'Taken'],
                              normalize=normalize, title=name + ' Predictor')
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred)
    
    classes = [classes[i] for i in unique_labels(y_true, y_pred)]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()