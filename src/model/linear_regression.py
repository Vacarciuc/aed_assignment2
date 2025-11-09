import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt


def get_linear_regression(data, predictors, predict):
    train_size = int(len(data) * 0.7)
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]

    x_train = train[predictors]
    y_train = train[predict]

    x_test = test[predictors]
    y_test = test[predict]

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Evaluare
    print("Linear Regression R^2: {:.3f}".format(r2_score(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("Coeficienți:", model.coef_)
    print("Intercept:", model.intercept_)
    print('=================================================================')

    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True)
    plt.title("Residuals linear regression")
    plt.show()
    return y_test, y_pred,