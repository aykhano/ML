
"""
Lojistik Regresyon ilə Diyabet Tahmini

İş Problemi:
Verilən xüsusiyyətlər əsasında şəxslərin diyabet xəstəsi olub-olmadığını 
proqnozlaşdıra biləcək bir maşın öyrənməsi modeli hazırlaya bilərsinizmi?

Veri Seti:
Veri seti, ABŞ Milli Diyabet–Həzm–Böyrək Xəstəlikləri İnstitutunda 
saxlanılan geniş məlumat bazasının bir hissəsidir. Məlumatlar, 
ABŞ-ın Arizona ştatının ən böyük 5-ci şəhəri olan Phoenix-də yaşayan 
21 yaş və üzəri Pima Indian qadınları üzərində aparılan diyabet 
araşdırmasına aiddir. Dataset 768 müşahidədən və 8 ədədi müstəqil 
dəyişəndən ibarətdir. Hədəf dəyişəni “outcome” adlanır və:

1: Diyabet test nəticəsinin pozitiv olması

0: Diyabet test nəticəsinin negativ olması

Dəyişənlər:

Pregnancies: Hamiləlik sayı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigree Function: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)
"""

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns 

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate

#Exploratory Data Analysis

df = pd.read_csv("datasets/diabetes.csv")

# Targetin Analizi

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
# plt.show()

100 * df["Outcome"].value_counts() / len(df)

#Feature analysis

df.describe().T

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
# print(plt.show())

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

# Example usage: plot the 'BloodPressure' distribution
plot_numerical_col(df, "BloodPressure")

for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

#Target vs Features

df.groupby("Outcome").agg({"Pregnancies": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in cols:
    target_summary_with_num(df, "Outcome", col)\
    
#Data preproccessing

df.shape
df.head()

df.isnull().sum()

print(df.describe().T)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

# MODEL & PREDICTION

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[:10]
y[:10]

#Model Evaluation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".Of")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

# Model Validation: Holdout

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))


