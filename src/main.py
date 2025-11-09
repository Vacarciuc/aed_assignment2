from get_data.get_data import read_file
from get_data.get_data import get_zero_code
from get_data.draw import draw_heat_map
from get_data.draw import draw_scatter_plot
from get_data.draw import plot_actual_vs_predicted
from Dataset import Dataset
import pandas as pd
from model.linear_regression import get_linear_regression
from model.non_linear_regression import non_linear_regression
from model.polynomial import get_polynomial_regression
from get_data.draw import plot_two_lines


def main(dataset: Dataset):
    predictors = []
    used_columns = []
    predict = ''
    compare_indicators = []

    if dataset == Dataset.FIRST:
        path = "../../data/dataset_1.xlsx"
        used_columns = ['Time', 'namq_10_gdp', 'namq_10_exi', 'tipsna40']
        predictors = ['namq_10_exi', 'tipsna40']
        predict = 'namq_10_gdp'
        compare_indicators = ['namq_10_gdp', 'tipsna40']
    elif dataset == Dataset.SECOND:
        path = "../../data/dataset_2.xlsx"
        used_columns = ['Time', 'prc_hicp_manr', 'irt_st_m', 'FP.CPI.TOTL.ZG']
        predictors = ['irt_st_m', 'FP.CPI.TOTL.ZG']
        predict = 'prc_hicp_manr'
        compare_indicators = ['prc_hicp_manr', 'irt_st_m']
    elif dataset == Dataset.THIRD:
        path = "../../data/dataset_3.xlsx"
        used_columns = ['Time', 'earn_nt_net', 'nama_10_lp_ulc', 'namq_10_gdp']
        predictors = ['nama_10_lp_ulc', 'namq_10_gdp']
        predict = 'earn_nt_net'
        compare_indicators = ['namq_10_gdp', 'earn_nt_net']
    elif dataset == Dataset.FIRST_NEW:
        path = "../../data/dataset_1New.xlsx"
        used_columns = ['Time', 'namq_10_gdp', 'namq_10_exi', 'CLVMNACSCAB1GQHU']
        predictors = ['namq_10_exi', 'CLVMNACSCAB1GQHU']
        predict = 'namq_10_gdp'
        compare_indicators = ['namq_10_gdp', 'CLVMNACSCAB1GQHU']
    else:
        return

    data = read_file(path, started=2000, finished=2024)

    missing_cols = [col for col in used_columns if col not in data.columns]
    if missing_cols:
        print(f"Coloanele lipsă în dataset: {missing_cols}")
        return

    data = data[used_columns]
    data['date'] = pd.to_datetime(data['Time'])
    data = data.set_index('date')
    quarterly_data = data.resample('Q').mean()
    quarterly_data.dropna(inplace=True)
    draw_heat_map(quarterly_data)
    data_zero = get_zero_code(data)
    plot_two_lines(data_zero, compare_indicators[0], compare_indicators[1])
    draw_scatter_plot(data_zero)
    draw_value1 = get_linear_regression(quarterly_data, predictors, predict)
    y_test, y_pred = draw_value1
    plot_actual_vs_predicted(y_test, y_pred, title="Linear Regression")
    draw_value2 = non_linear_regression(quarterly_data, predictors, predict)
    y_test2, y_pred2 = draw_value2
    plot_actual_vs_predicted(y_test2, y_pred2, title="Non-Linear Regression")
    draw_value3 = get_polynomial_regression(quarterly_data, predictors, predict)
    y_test3, y_pred3 = draw_value3
    plot_actual_vs_predicted(y_test3, y_pred3, title="Polynomial Regression")


if __name__ == "__main__":
    main(Dataset.FIRST_NEW)