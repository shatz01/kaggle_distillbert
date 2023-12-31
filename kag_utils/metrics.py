from sklearn.metrics import roc_auc_score

def compute_roc_auc(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if labels.std() < 1E-8: # only one class present in dataset
        return {"roc_auc": 0.0}
    ps = softmax(logits, axis=-1)[:,1]
    return {"roc_auc": roc_auc_score(labels, ps)}
