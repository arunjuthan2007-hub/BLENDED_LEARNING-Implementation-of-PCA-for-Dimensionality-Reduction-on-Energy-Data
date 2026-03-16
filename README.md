# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import pandas, PCA, StandardScaler, and matplotlib.

2. Load Data: Load the dataset using pandas.read_csv().

3. Select Features: Choose 'Height(Inches)', 'Weight(Pounds)', 'Weight(Kilograms)' for PCA.

4. Scale Data: Standardize the features with StandardScaler.

5. Perform PCA: Apply PCA to reduce the data to 2 components.

6. Create PCA DataFrame: Convert the PCA output into a DataFrame.

7. Visualize Data: Plot the data in 2D using a scatter plot.

8. Explain Variance: Print the explained variance ratio for each component and the total variance.

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Arunjuthan.M.A
RegisterNumber: 212225230020
*/
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("HeightsWeights.csv")
print(data.head())
print(data.columns)
X = data[['Height(Inches)', 'Weight(Pounds)']]  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
print("\nName: Arunjuthan.M.A")
print("Reg No: 212225230020\n")
print("Explained Variance Ratio for each Principal Component:", explained_variance)
print("Total Explained Variance:", sum(explained_variance))
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Heights and Weights Dataset")
plt.show()
```

## Output:



## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
