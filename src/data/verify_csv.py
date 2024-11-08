import pandas as pd
import os

def verify_csv():
    csv_path = './data/sample/semantic_drone/class_dict_seg.csv'
    df = pd.read_csv(csv_path, skipinitialspace=True)
    print("CSV columns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)

if __name__ == '__main__':
    verify_csv() 