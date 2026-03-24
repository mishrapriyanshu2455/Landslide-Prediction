
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv("synthetic_landslide_dataset.csv")



X = df.drop('landslide', axis=1)
y = df['landslide']


# 'stratify=y' ensures the ~33% landslide ratio is perfectly maintained in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
}
mod= Pipeline([
    
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced' 
    ))
])

grid = GridSearchCV(
    mod,
    param_grid,
    cv=5,
    scoring='recall',   
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
mod = grid.best_estimator_





print("Training model...")
mod.fit(X_train, y_train)


y_pred =mod.predict(X_test)


y_prob = mod.predict_proba(X_test)[:,1]


y_pred = (y_prob > 0.35).astype(int)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


rf_model = mod.named_steps['classifier']
importances = rf_model.feature_importances_


plt.figure(figsize=(10, 6))

indices = np.argsort(importances)

plt.barh(range(len(indices)), importances[indices], color='steelblue', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importance for Landslide Prediction')
plt.tight_layout()
plt.show()
joblib.dump(mod, "landslide_model.pkl")


joblib.dump(mod, "landslide_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
