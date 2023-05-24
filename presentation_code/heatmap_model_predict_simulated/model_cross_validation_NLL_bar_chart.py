import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


example_NLLs = [46.78653438984054, 55.78840895784988, 
                53.516841506447385, 60.313311287583524, 58.91318398912639, 55.07415860394206, 61.80344442870653, 69.96287000963532]

distance_NLLs = [209.3421981,  244.38414186, 309.07730689, 282.09366861]
head_angle_NLLs = [210.88248726, 256.88697357, 299.59442668, 305.00875267]

current_NLLs = head_angle_NLLs

bar_titles = [f"Model {i + 1}" for i in range(len(current_NLLs))]

df = pd.DataFrame({"Model number":bar_titles, "Negative log-likelihoods":current_NLLs})


sns.set(font_scale=1.4)
sns.set_style('whitegrid', {"axes.spines.top":False, "axes.spines.right":False, "axes.yaxis.grid":False})
fig, p1 = plt.subplots(figsize=(8,6))
# p1.bar(bar_titles, current_NLLs, color='orange')
p1 = sns.barplot(df, x="Model number", y="Negative log-likelihoods")


p1.set_title("Model cross-validated negative log-likelihoods", fontsize=18)
p1.set_xlabel("Model Number", fontsize=14)
p1.set_xlabel("Negative Log Likelihoods", fontsize=14)
p1.axes.xaxis.grid(False)

for i, v in enumerate(current_NLLs):
    p1.text(i, v+1, f"{v:.2f}", ha='center', fontsize=12)

p1.tick_params(axis='x', labelsize=12)

plt.show()