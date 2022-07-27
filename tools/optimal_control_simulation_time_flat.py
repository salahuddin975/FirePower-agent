import pandas as pd

file = f"perfect_control_simulation_time.csv"
df = pd.read_csv(file)

all_val = []
for val in df.values.tolist():
    all_val += val[:300]


df = pd.DataFrame(columns=["all_time"])
df["all_time"] = all_val
df.to_csv(f"perfect_control_simulation_all_time.csv")