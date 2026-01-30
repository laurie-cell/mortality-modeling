import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from config import MAX_ITER, MODEL_PATH, LOGISTIC_CONFIGS
from scripts.preprocess import preprocess_data

# Load data
X_train, X_test, y_train, y_test = preprocess_data()

best_model = None
best_auc = -1
best_model_name = None

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

for name, params in LOGISTIC_CONFIGS.items():
    print(f"\nTraining Logistic Regression: {name}")

    model = LogisticRegression(
        max_iter=MAX_ITER,
        class_weight=class_weight_dict,
        **params
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC ({name}): {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(
    {
        "model": best_model,
        "model_name": best_model_name,
        "features": X_train.columns.tolist(),
        "config": LOGISTIC_CONFIGS[best_model_name]
    },
    MODEL_PATH
)

print(f"\nBest model: {best_model_name}")
print(f"Saved to {MODEL_PATH}")
