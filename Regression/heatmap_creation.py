import seaborn as sns
import numpy as np
# Creating a random dataset
data = np.random.rand(5, 5)
# Creating a heatmap
sns.heatmap(data, annot=True, cmap="coolwarm")
plt.show()