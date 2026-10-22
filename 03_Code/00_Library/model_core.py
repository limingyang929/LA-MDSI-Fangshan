import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, matthews_corrcoef, balanced_accuracy_score
def estimator(name: str, seed: int):
    """Use fixed legacy hyperparameters with one model thread and no tuning."""
    if name == 'logistic':
        return LogisticRegression(C=1.0, max_iter=4000, solver='liblinear', random_state=seed)
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=300,max_depth=20,min_samples_leaf=3,max_features='sqrt',random_state=seed,n_jobs=1)
    if name == 'rbf_svm':
        return SVC(C=0.5,gamma=0.1,probability=True,random_state=seed,cache_size=1000)
    if name == 'xgboost':
        return XGBClassifier(n_estimators=500,learning_rate=0.1,max_depth=5,min_child_weight=8,gamma=1,subsample=0.7,colsample_bytree=0.7,reg_alpha=0.5,reg_lambda=8,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=seed,n_jobs=1)
    raise ValueError(name)

def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Evaluate uncompressed validation observations under the sampled binary target."""
    clipped = np.clip(p, 1e-6,1-1e-6)
    result = {'roc_auc':float(roc_auc_score(y,p)), 'pr_auc':float(average_precision_score(y,p)), 'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,clipped)), 'mcc':float(matthews_corrcoef(y,p>=0.5)), 'balanced_accuracy':float(balanced_accuracy_score(y,p>=0.5))}
    for bins in (5,10,20):
        assignment = np.minimum((np.clip(p,0,1)*bins).astype(int),bins-1)
        result[f'ece_equal_width_{bins}'] = float(sum(abs(p[assignment==j].mean()-y[assignment==j].mean())*(assignment==j).mean() for j in range(bins) if np.any(assignment==j)))
    return result
