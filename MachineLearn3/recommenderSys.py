import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

columns_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("u.data", sep="\t", names=columns_names)
movie_titles = pd.read_csv("Movie_Id_Titles")

df = pd.merge(df, movie_titles, on="item_id")

sns.set_style("white")
print(df.groupby("title")["rating"].mean().sort_values(ascending=False).head())

ratings = pd.DataFrame(df.groupby("title")["rating"].mean())
ratings["num of ratings"] = pd.DataFrame(df.groupby("title")["rating"].count())

ratings["num of ratings"].hist(bins=70)
plt.show()

ratings["rating"].hist(bins=70)
plt.show()

sns.jointplot(data=ratings, x="rating", y="num of ratings", alpha=.5)
plt.show()

moviemat = df.pivot_table(index="user_id", columns="title", values="rating")

starwars_user_ratings = moviemat["Star Wars (1977)"]
liarliar_user_ratings = moviemat["Liar Liar (1997)"]

sim_to_star = moviemat.corrwith(starwars_user_ratings)
sim_to_liar = moviemat.corrwith(liarliar_user_ratings)
corr_star = pd.DataFrame(sim_to_star, columns=["Correlation"])
corr_star.dropna(inplace=True)

corr_star = corr_star.join(ratings["num of ratings"])
corr_star[corr_star["num of ratings"] > 100].sort_values("Correlation", ascending=False)

corr_liar = pd.DataFrame(sim_to_liar, columns=["Correlation"])
corr_star.dropna(inplace=True)

corr_liar = corr_liar.join(ratings["num of ratings"])
corr_liar[corr_liar["num of ratings"] > 100].sort_values("Correlation", ascending=False)
