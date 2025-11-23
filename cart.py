# Decision Tree Classification: CART

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model


import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

# 1. EXPLATORY Data Analysis

# 2. Data Processing & Feature Engineering

# 3. Modeling using CART

df = pd.read_csv("datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion y_pres for matrix:
y_pred = cart_model.predict(X)

# AUC ucun y_prob:
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion matrix
print(classification_report(y, y_pred))

# Auc
roc_auc_score(y, y_prob)

# Holdout 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Issue
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# Test Issue
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

# CV ile basari degerlendirme 

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model, 
                            X, y, 
                            cv=10, 
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Hyperparameter Optimization with GridSearchCV

cart_model.get_params()

cart_params = {'max_depth': range(1, 11), 
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model, 
                              cart_params,
                              #scoring="roc_auc", 
                              cv=5, 
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_
cart_best_grid.best_score_

random = X.sample(1, random_state=45)
cart_best_grid.predict(random)

# 5. Final model

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()

cart_final = cart_final.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_model, 
                            X, y, 
                            cv=5, 
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 6. Feature Importance

cart_final.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

print(plot_importance(cart_final, X, num=5))

# 7. Analyzing Model Complexity with Learning Curves (BONUS)

train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth", 
                                           param_range=range(1, 11),
                                           scoring="roc_auc", 
                                           cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)

plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score, 
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv
    )

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')
    
    plt.plot(param_range, mean_test_score, 
             label="Validation Score", color='g')
    
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

print(val_curve_params(cart_model, X, y, "max_depth", range(1, 11)))