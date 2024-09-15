import csv 
import pandas as pd

file_path = "./pest_output.csv"

df = pd.read_csv(file_path)

df = df.groupby(["point"])

for point, group in df:
    print(f"Point {point} has {len(group)} observations")

