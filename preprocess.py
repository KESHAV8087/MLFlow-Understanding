import pandas as pd

# Load both datasets, both using the semicolon separator.
red_wine = pd.read_csv("data/winequality-red.csv", sep=";")
white_wine = pd.read_csv("data/winequality-white.csv", sep=";")

# Add an indicator column so the model knows which type each wine is.
red_wine["is_red"] = 1
white_wine["is_red"] = 0

# Combine the two datasets into one.
data = pd.concat([red_wine, white_wine], axis=0)

# Clean column names: replace spaces with underscores so no library complains later.
data.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

# Define the target: a wine is "high quality" if its quality score is 7 or above.
# This turns the problem into classification: high quality (1) or not (0).
data["high_quality"] = (data["quality"] >= 7).astype(int)

# Save the processed dataset so later steps can reuse it.
data.to_csv("data/processed_wine.csv", index=False)

# Print summary information to verify the preprocessing worked.
print("Combined shape:", data.shape)
print("\nColumns after cleaning:")
print(list(data.columns))
print("\nHigh quality distribution (0 = not high, 1 = high):")
print(data["high_quality"].value_counts())
print("\nProportion high quality:", round(data["high_quality"].mean(), 3))