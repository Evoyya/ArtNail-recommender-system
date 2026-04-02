import os
from pathlib import Path
import pandas as pd


def build_year_user_items():
    
    project_root = Path(__file__).resolve().parents[3]
    print(f'Корневая папка: {project_root}')

    base_dir = project_root / "Tables" / "BaseTable"
    out_dir = project_root / "Tables" / "BaseTable"
    out_dir.mkdir(parents=True, exist_ok=True)

    filenames = [
        "User_items_04_03_2025_04_06_2025.csv",
        "User_items_04_07_2025_04_08_2025.csv",
        "User_items_04_10_2025_04_12_2025.csv",
        "User_items_05_06_2025_03_07_2025.csv",
        "User_items_05_08_2025_03_10_2025.csv",
        "User_items_05_12_2025_04_03_2026.csv",
    ]

    dfs = []
    for fname in filenames:
        path = base_dir / fname
        df = pd.read_csv(path, encoding='cp1251', sep=';')
        dfs.append(df)

    df_year = pd.concat(dfs, ignore_index=True)

    
    out_path = out_dir / "User_items_year.csv"
    df_year.to_csv(out_path, index=False, sep=',')

    print(f"Склеено файлов: {len(dfs)}")
    print(f"Итоговая таблица: {df_year.shape[0]} строк, {df_year.shape[1]} столбцов")
    print(f"Сохранено в: {out_path}")

    return df_year

def print_info_df_year(df_year):
    print(df_year.info())

if __name__ == "__main__":
   df_year = build_year_user_items()
   print_info_df_year(df_year)





