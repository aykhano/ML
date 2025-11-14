#Sales Prediction with linear Regression

import pandas as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("datasets/advertising.csv")

X = df[["TV"]]
y = df[["Sales"]]

#Model

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w * TV

# sabit (b - bias)
reg_model.intercept_[0]

#tv nin katsayisi(w1)
reg_model.coef_[0][0]

#Tahmin

# 150 birimlik tv harcamasi olsa ne kadar satish olmasi gozlenilir?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# 500 birimlik tv harcamasi olsa ne kadar saish olur?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

df.describe().T

# Modelin Gorselleshdirilmesi

g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's':9},
                ci=False, color="r")
g.set_title(f"Model denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV * {round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satish sayisi")
g.set_xlabel("Tv Hacamalari")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

# Tahmin Basarisi

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-KARE
reg_model.score(X, y)

#Multiple Linear Regression

df = pd.read_csv("datasets/advertising.csv")

X = df.drop('sales', axis=1)

y = df[["Sales"]]

#Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)