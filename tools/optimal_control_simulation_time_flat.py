import pandas as pd

file = f"/Users/smsalahuddinkadir/Desktop/test_results/perfect_control/time_limit_900/perfect_control_simulation_time.csv"
df = pd.read_csv(file)

all_val = []
for row in df.values.tolist():
    all_val += row[:100]

df = pd.DataFrame(columns=["all_time"])
df["all_time"] = all_val
df.to_csv(f"perfect_control_simulation_all_time.csv")