import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

sns.set_style("whitegrid")

titanic = sns.load_dataset("titanic")
titanic.head()

# EXERCISE 1
sns.jointplot(x='fare', y='age', data=titanic)
plt.show()

# EXERCISE 2
sns.displot(titanic['fare'], kde=False, color="red", bins=30)
plt.show()

# EXERCISE 3
sns.boxplot(x="class", y="age", data=titanic, palette="rainbow")
plt.show()

# EXERCISE 4
sns.swarmplot(x="class", y="age", data=titanic, palette="Set2")
plt.show()

# EXERCISE 5
sns.countplot(x="sex", data=titanic)
plt.show()

# EXERCISE 6
sns.heatmap(titanic.corr(), cmap="coolwarm")
plt.title("titanic.corr()")
plt.show()

# EXERCISE 7
g = sns.FacetGrid(data=titanic, col="sex")
g.map(plt.hist, 'age')
plt.show()
