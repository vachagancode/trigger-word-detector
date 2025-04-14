import os
import csv
import pandas as pd

with open("base_annotations_file.csv", "a", newline='') as f:
    writer = csv.writer(f)
    fields = ["name", "path", "label"]
    writer.writerow(fields)

    # Get data from raw_data folder
    labels = ["positive", "negative"]
    base_path = "./raw_data"
    for label in labels:
        dir = f"{base_path}/{label}"
        files = os.listdir(dir)
        for file in files:
            file_path = f"{base_path}/{label}/{file}"
            writer.writerow([file, file_path, label])
    

# now shuffle the annotations file
df = pd.read_csv("./base_annotations_file.csv")
shuffled = df.sample(frac=1)
shuffled.to_csv("./annotations_file.csv")
print("Annotations file initialized successfully")