import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

def read_parquet_file(file_path: str) -> pd.DataFrame:
    table = pq.read_table(file_path)
    return table.to_pandas()
path_1h = "data/processed/returns_1h.parquet"

df_1h = read_parquet_file(path_1h)
print(df_1h.info())

df_1d = read_parquet_file("data/processed/returns_1d.parquet")
print(df_1d.info())