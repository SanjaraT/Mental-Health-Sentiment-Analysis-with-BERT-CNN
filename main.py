import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Combined Data.csv")
# print(df.head())

#Data Distribution
plt.figure(figsize=(6,5))
sns.countplot(x=df['status'])
plt.xticks(rotation=45)
plt.title("Class Distribution")
# plt.show()

df = df.dropna(subset=['statement'])


#Data Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.strip()

df['clean_text'] = df['statement'].apply(clean_text)
print(df.head())

#label Encoding
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['status'])

num_classes = len(le.classes_)
print(le.classes_)


