import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def draw_heat_map(quarterly_data):
    sns.set(style="darkgrid")
    quarterly_data.drop(columns=["Time"], inplace=True)
    sns.heatmap(quarterly_data.corr(), annot=True, cmap='coolwarm')
    plt.show()

def draw_scatter_plot(data):
    print(data.columns)
    # Asumăm că dataframe-ul se numește quarterly_data
    data['Time'] = pd.to_datetime(data['Time'])
    plt.figure(figsize=(12, 6))

    x = data['Time']  # acum este datetime

    for col in data.columns:
        if col != 'Time':
            plt.scatter(x, data[col], label=col, alpha=0.7)

    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_two_lines(data, col1, col2, title="Compararea a două variabile"):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data[col1], label=col1, color='blue', linewidth=2)
    plt.plot(data['Time'], data[col2], label=col2, color='orange', linewidth=2)

    plt.title(title)
    plt.xlabel("Timp")
    plt.ylabel("Valoare")
    plt.legend()
    plt.grid(True)
    plt.show()
