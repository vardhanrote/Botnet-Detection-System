from pathlib import Path
import pandas as pd


def load_csv(file_path):
    """
    Load a CSV dataset.

    Parameters
    ----------
    file_path : str or Path
        Location of the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    print(f"Loading dataset: {file_path}")

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


def show_dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("\n========== DATASET INFORMATION ==========")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())