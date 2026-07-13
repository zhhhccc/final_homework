import pyarrow.parquet as pq
import pandas as pd

#读取数据，转换成pandas文件，并保存为pkl
trips = pq.read_table('D:/tasks/final/data/yellow_tripdata_2026-01.parquet')
trips = trips.to_pandas()
df=pd.DataFrame(trips)
df.to_pickle("D:/tasks/final/data.pkl")