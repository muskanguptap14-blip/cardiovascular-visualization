import pandas as pd

df = pd.read_csv(r"C:\Users\muska\Downloads\extention+of+z+alizadeh+sani+dataset\extention of Z-Alizadeh sani dataset.csv")

#inspect data structure 
print("shape:",df.shape)
print("first five rows of the dataset:")
print("head:",df.head())
print("\nData  types")
print("data types:",df.dtypes)
print("\nDataframe info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Target distribution
print("\nCAD distribution:")
print(df["Cath"].value_counts(dropna=False))

# Duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Numeric summary
print("\nStatistical summary:")
print(df.describe().T)

print("\nAll columns:")
print(df.columns.tolist())

print("\nTarget column: Cath")
print(df["Cath"].value_counts(dropna=False))

print("\nMissing values in all columns:")
print(df.isnull().sum().to_string())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nNon-numeric columns:")
print(df.select_dtypes(exclude="number").columns.tolist())
