import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def combine_mode_bl_type (x):
  if pd.isnull (x ['bl_type']):
    return x ['mode']
  else:
    return '_'.join ([x ['mode'], x ['bl_type']])

df_avg = pd.read_csv("z_score_avg.csv")
sns.set_theme(style="whitegrid")
rs = np.random.RandomState(4)
df_clean = df_avg.drop(df_avg[(df_avg["mode"] == "new") | (df_avg["mode"] == "no")].index)

df_clean['mode_bl_type'] = df_clean.apply(combine_mode_bl_type, axis=1)

# 创建一个分面网格图对象，按照gamma值分列绘制子图
# sns.lineplot(x="delta", y="true_positive", style="logic", markers=["o", "s", "d", "^"], data=df_clean)

g = sns.FacetGrid(df_clean, col="gamma", hue="mode_bl_type", palette="Set1", hue_kws={"marker": ["P", "s", "^", "v"], "markersize": [10, 8, 8, 8], "markeredgewidth": [0, 0, 0, 0], "alpha": [0.8, 0.8, 0.8, 0.8]})
g.map(sns.lineplot, "delta", "true_positive", marker=True, dashes=False)

g.set_axis_labels("Delta", "True Positive")
g.set_titles("Gamma = {col_name}")

g.add_legend()

plt.savefig("llama2.pdf", dpi=400)
plt.show()

# Run this cell to print subframes of a groupby object, no further action is needed
'''for gamma, group in df_avg.groupby("gamma"):
    print(f"Gamma: {gamma}") 
    display(group)'''