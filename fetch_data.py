import urllib.request

# The UCI Machine Learning Repository hosts the wine quality datasets publicly.
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Download each file into the data folder.
urllib.request.urlretrieve(red_url, "data/winequality-red.csv")
urllib.request.urlretrieve(white_url, "data/winequality-white.csv")

print("Downloaded both wine datasets into the data folder.")