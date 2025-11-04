from get_data.get_data import read_file
from Dataset import Dataset


def main(dataset: Dataset):
    if dataset == Dataset.FIRST:
        path = "../../data/dataset_1.xlsx"
    elif dataset == Dataset.SECOND:
        path = "../../data/dataset_2.xlsx"
    elif dataset == Dataset.THIRD:
        path = "../../data/dataset_3.xlsx"
    else:
        return

    data = read_file(path, started=2000, finished=2024)
    print(data)

if __name__ == "__main__":
    main(Dataset.FIRST)