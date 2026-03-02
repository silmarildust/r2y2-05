import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("noise_levels.csv")

betti_df = (
    df.groupby(["level_value", "replicate", "dimension"])
      .size()
      .reset_index(name="betti")
)

betti_pivot = betti_df.pivot_table(
    index=["replicate", "dimension"],
    columns="level_value",
    values="betti",
    fill_value=0
)


noise_levels = sorted(df["level_value"].unique())
dimensions = sorted(df["dimension"].unique())


colors = {
    0: '#013366',   # H0
    1: '#ea7334',   # H1
    2: '#008b8b'    # H2
}


fig, ax = plt.subplots(figsize=(8, 6))


for dim in dimensions:


    dim_data = betti_pivot.loc[
        betti_pivot.index.get_level_values("dimension") == dim
    ]


    all_reps = dim_data.values

    for y in all_reps:
        ax.plot(noise_levels, y,
                color=colors.get(dim, 'gray'),
                alpha=0.35,
                marker='o')

    mean_y = np.mean(all_reps, axis=0)
    std_y = np.std(all_reps, axis=0)
    ci = 1.96 * (std_y / np.sqrt(len(all_reps)))


    ax.plot(noise_levels, mean_y,
            color=colors.get(dim, 'gray'),
            linewidth=2,
            label=f'$H_{dim}$ Mean')


    ax.fill_between(noise_levels,
                    mean_y - ci,
                    mean_y + ci,
                    color=colors.get(dim, 'gray'),
                    alpha=0.2)


ax.set_xlabel("Noise Level")
ax.set_ylabel("Betti Numbers")
ax.set_title("Betti Numbers vs Noise Level")
ax.legend()
plt.tight_layout()
plt.show()



