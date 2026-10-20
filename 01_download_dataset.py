import kagglehub
import os

# Download the competition dataset
path = kagglehub.competition_download("name-that-llm")

print("Competition files downloaded successfully!")
print("Dataset location:", path)

# Display the downloaded files
print("\nFiles available:")

for file in os.listdir(path):
    print(file)