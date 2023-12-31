from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import numpy as np

def compute_roc_auc(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if labels.std() < 1E-8: # only one class present in dataset
        return {"roc_auc": 0.0}
    ps = softmax(logits, axis=-1)[:,1]
    return {"roc_auc": roc_auc_score(labels, ps)}

def compute_accuracy(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if labels.std() < 1E-8: # only one class present in dataset
        return {"accuracy": 0.0}
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}