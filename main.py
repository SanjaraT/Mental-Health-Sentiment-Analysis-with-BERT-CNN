import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Combined Data.csv")
print(df.head())


#Data Distribution
plt.figure(figsize=(6,5))
sns.countplot(x=df['status'])
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.show()

