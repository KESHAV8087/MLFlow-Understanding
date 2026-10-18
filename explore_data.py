import pandas as pd

# Both UCI wine files use a semicolon as the separator.
red_wine = pd.read_csv("data/winequality-red.csv", sep=";")
white_wine = pd.read_csv("data/winequality-white.csv", sep=";")

# Check the shape, meaning number of rows and columns, of each dataset.
print("Red wine shape:", red_wine.shape)
print("White wine shape:", white_wine.shape)

# Show the column names so we know what features we have.
print("\nColumns:")
print(list(red_wine.columns))

# Show the first few rows of the red wine data to confirm it loaded correctly.
print("\nFirst few rows of red wine:")
print(red_wine.head())

# Check the target column, quality, to see its range of values.
print("\nQuality value counts (red wine):")
print(red_wine["quality"].value_counts().sort_index())