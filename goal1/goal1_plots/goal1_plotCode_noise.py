import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("noise_levels.csv")

df = df.dropna(subset=["dimension", "level_value", "length"])

df["dimension"] = df["dimension"].astype(str)
df["length"] = pd.to_numeric(df["length"], errors="coerce")
df = df.dropna(subset=["length"])

levels_to_plot = [20, 35, 50, 65, 80]
df = df[df["level_value"].isin(levels_to_plot)].copy()

df["level_value"] = pd.Categorical(df["level_value"], categories=levels_to_plot, ordered=True)

# Styling
sns.set_theme(style="whitegrid", font_scale=1.1, rc={"axes.spines.top": False})

palette = {"0": "#66c2a5", "1": "#fc8d62", "2": "#8da0cb"}

# Plot
g = sns.displot(
    data=df,
    x="length",
    hue="dimension",
    row="level_value",
    kind="hist",
    multiple="dodge",
    bins=30,
    binrange=(0, 10),  
    height=1.8,
    aspect=3,
    palette=palette,
    edgecolor="black",
    linewidth=0.3,
    alpha=0.85,
    common_bins=True,
    common_norm=False
)

g.set(xlim=(0, 10), ylim=(0, 500))  

g.set_axis_labels("Length", "")
g.set_titles(row_template="Noise Level = {row_name}")
g.fig.subplots_adjust(top=0.88, hspace=0.4)
g.fig.suptitle("Persistence Distributions\nBarcode Lengths by Noise Level", fontsize=16, y=0.98)

if g._legend is not None:
    g._legend.set_title("Dimension")

plt.show()
