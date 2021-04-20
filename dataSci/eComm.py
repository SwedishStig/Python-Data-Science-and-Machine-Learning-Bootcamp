import pandas as pd
ecom = pd.read_csv('Ecommerce Purchases')

print(ecom.head())

print(ecom.value_counts())
print(len(ecom.columns))

print(ecom['Purchase Price'].mean())

print(ecom['Purchase Price'].max())
print(ecom['Purchase Price'].min())

print(len(ecom[ecom['Language'] == "en"]))

print(len(ecom[ecom['Job'] == "Lawyer"]))

print(ecom["AM or PM"].value_counts())

print(ecom['Job'].value_counts().head(5))

print(ecom[ecom['Lot'] == "90 WT"]["Purchase Price"])

print(ecom[ecom['Credit Card'] == 4926535242672853]["Email"])

print(len(ecom[(ecom['CC Provider'] == "American Express") & (ecom["Purchase Price"] > 95)]))

print(sum(ecom['CC Exp Date'].apply(lambda ex: ex[3:] == "25")))

print(ecom['Email'].apply(lambda em: em.split('@')[1]).value_counts().head(5))
