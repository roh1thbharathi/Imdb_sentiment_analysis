import pandas as pd

# Path to the CSV file
csv_path = "data/IMDB Dataset.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Print basic info
print("Shape (rows, columns):", df.shape)
print("Column names:", df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3))

print("\nSample review (first 400 chars):")
print(df['review'][0][:400])

print("\nSample label:")
print(df['sentiment'][0])