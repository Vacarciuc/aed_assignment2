import os
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler


def read_file(
    path: str,
    started: int = None,
    finished: int = None
) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_excel(file_path, sheet_name="data", header=None)
    df.columns = df.iloc[0]
    df = df.drop(0)
    # Setăm coloana 'Indicator' ca index și transpunem
    df = df.set_index('Indicator').T
    # Resetăm index-ul numeric într-o coloană 'time'
    df = df.reset_index().rename(columns={'index': 'time'})
    # Redenumește coloana Indicator în Time
    df = df.rename(columns={0: 'Time'})
    #############
    # Asigurăm că Time e datetime
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

    # aplicăm filtrarea după ani dacă se specifică
    if started and finished:
        mask = (df['Time'].dt.year >= started) & (df['Time'].dt.year <= finished)
        df = df.loc[mask]
        print(f"📅 Filtrat pentru perioada {started} - {finished} ({len(df)} rânduri)")

    #aplicarea filtrelor de curatare
    #detectarea si stergerea valorilor null/lipsa
    df.isnull().sum()
    df.dropna(inplace=True)
    # Verifică coloanele finale
    print('Columns', df.columns)
    return df


def get_zero_code(data: DataFrame):
    scaler = StandardScaler()
    # Elimină coloanele nenumerice (Indicator, Time)
    numeric_data = data.drop(columns=["Time"], errors="ignore")
    # Aplică scalarea doar pe valori numerice
    scaled = scaler.fit_transform(numeric_data)
    # Reconstruiește DataFrame-ul scalat
    scaled_df = pd.DataFrame(
        scaled,
        columns=numeric_data.columns
    )
    scaled_df["Time"] = data["Time"].values
    return scaled_df
