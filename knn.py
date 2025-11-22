# KNN

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model


import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)

# 1. EDA
df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()


# 2. Data-Preprocessing & Feature engineering

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_scaled = StandardScaler().fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# 3. Modeling & Prediction

knn_model = KNeighborsClassifier().fit(X, y)
random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

# Model Evaluation

# Confusion matrix y_pred:
y_pred = knn_model.predict(X)

# AUC icin y_prob:
y_prob = knn_model.predict(X)[:, 1]

print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 1. Sample size can be increased.
# 2. Data preprocessing
# 3. Feature engineering
# 4. Optimizations can be made for the relevant algorithm.

knn_model.get_params()

# Hyperparametr Optimization

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

# Final Model

knn_model = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

random_user = X.sample(1)

