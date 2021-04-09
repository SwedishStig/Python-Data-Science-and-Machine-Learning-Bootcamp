import pandas as pd
import matplotlib.pyplot as plt
# import df3 as beans
df3 = pd.read_csv('df3')
# %matplotlib inline

# plt.show after every plot command

# EXERCISE 1
df3.plot.scatter(x='a', y='b', color="red", figsize=(12, 3), s=50)
plt.show()

# EXERCISE 2
df3['a'].plot.hist()
plt.show()

# EXERCISE 3
plt.style.use("ggplot")
df3['a'].plot.hist(bins=20, alpha=.6)
plt.show()

# EXERCISE 4
df3[['a', 'b']].plot.box()
plt.show()

# EXERCISE 5
df3['d'].plot.kde(lw=4, ls='--')
plt.show()

# EXERCISE 6
# .ix is deprecated, need to use .iloc instead
# f = plt.figure()
df3.iloc[:30].plot.area(alpha=.6)
plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
plt.show()
