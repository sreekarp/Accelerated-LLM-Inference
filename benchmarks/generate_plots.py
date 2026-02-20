import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmark_results.csv")

# -------- Latency Graph --------
for mode in df["mode"].unique():

    subset = df[df["mode"] == mode]

    plt.plot(
        subset["prompt_length"],
        subset["latency"],
        marker="o",
        label=mode
    )

plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Latency (seconds)")
plt.title("Decoding Latency vs Prompt Length")
plt.legend()
plt.grid()

plt.savefig("latency_vs_prompt_length.png")
plt.show()


# -------- Speedup vs Greedy --------
greedy_df = df[df["mode"] == "greedy"].set_index("prompt_length")

speedup_data = []

for _, row in df.iterrows():
    if row["mode"] == "greedy":
        continue

    greedy_latency = greedy_df.loc[row["prompt_length"]]["latency"]
    speedup = greedy_latency / row["latency"]

    speedup_data.append({
        "prompt_length": row["prompt_length"],
        "mode": row["mode"],
        "speedup": speedup
    })

speedup_df = pd.DataFrame(speedup_data)

for mode in speedup_df["mode"].unique():

    subset = speedup_df[speedup_df["mode"] == mode]

    plt.plot(
        subset["prompt_length"],
        subset["speedup"],
        marker="o",
        label=mode
    )

plt.xlabel("Prompt Length (tokens)")
plt.ylabel("Speedup vs Greedy")
plt.title("Speedup Comparison")
plt.legend()
plt.grid()

plt.savefig("speedup_vs_prompt_length.png")
plt.show()
