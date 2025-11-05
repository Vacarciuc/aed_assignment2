import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

def get_polynomial_regression(data, predictors, predict, degree=2):
    train_size = int(len(data) * 0.7)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    # Crearea termenilor polinomiali
    poly = PolynomialFeatures(degree=degree)

    # Preprocesare pentru setul de date de antrenament
    x_train_poly = poly.fit_transform(train[predictors])
    y_train = train[predict]

    # Preprocesare pentru setul de date de test
    x_test_poly = poly.transform(test[predictors])
    y_test = test[predict]

    # Crearea și antrenarea modelului de regresie liniară pe termenii polinomiali
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    y_pred = model.predict(x_test_poly)

    # Evaluare
    print("Polynomial regression coefficients:", model.coef_)
    print("R2:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    print("Coeficienți:", model.coef_)
    print("Intercept:", model.intercept_)
    print("==================================================")

    # Graficul reziduurilor
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True)
    plt.title("Residuals polynomial")
    plt.show()

# Apelarea funcției
# get_polynomial_regression(data)
