import os
import csv

with open("annotations_file.csv", "a", newline='') as f:
    writer = csv.writer(f)
    fields = ["name", "path", "label"]
    writer.writerow(fields)

    # Get data from raw_data folder
    labels = ["positive", "negative", "background"]
    base_path = "./raw_data"
    for label in labels:
        dir = f"{base_path}/{label}"
        files = os.listdir(dir)
        for file in files:
            file_path = f"{base_path}/{label}/{file}"
            writer.writerow([file, file_path, label])

    print("Annotations file initialized successfully")