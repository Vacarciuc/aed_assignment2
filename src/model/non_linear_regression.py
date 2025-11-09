import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Modelul non-liniar pe care-l vom folosi (exponențial)
def model(x, a, b, c):
    une_rt_q, demo_pjan = x
    return a * np.exp(b * une_rt_q + c * demo_pjan)

def non_linear_regression(data, predictors, predict):
    # Divizarea în seturi de date
    train_size = int(len(data) * 0.7)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    # Normalizarea variabilelor (scalare)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(train[predictors])
    x_test_scaled = scaler.transform(test[predictors])

    # Variabilele țintă
    y_train = train[predict]
    y_test = test[predict]

    # Ajustarea modelului non-liniar cu curve_fit
    try:
        # Estimarea parametrilor inițiali
        popt, _ = curve_fit(model, x_train_scaled.T, y_train, p0=[1, 0.1, 0.1])
    except RuntimeError as e:
        print("Eroare la ajustarea modelului:", e)
        return

    # Predicțiile folosind parametrii ajustați
    y_pred = model(x_test_scaled.T, *popt)

    # Evaluare
    print("non linear regression coefficients:", popt)
    print("R2:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))

    print("Coeficienți ajustați:", popt)
    print("============================================")

    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True)
    plt.title("Residuals non-linear regression")
    plt.show()

    return y_test, y_pred


