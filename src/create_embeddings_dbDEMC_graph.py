import csv
import json

# File paths
mirna_csv_path = 'data/_miRNA_embeddings_dbDEMC.csv'
disease_csv_path = 'data/_disease_embeddings_dbDEMC_updated.csv'
relation_csv_path = 'data/_filtered_dbDEMC_Homo_sapiens.csv'
output_json_path = 'data/_miRNA_disease_embeddings_dbDEMC_v3.json'

# Function to read embeddings from a CSV file
def read_embeddings(file_path):
    embeddings = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip the header
        for row in reader:
            name = row[0]
            embedding = list(map(float, row[1:]))
            embeddings[name] = embedding
    return embeddings

# Read miRNA and disease embeddings
mirna_embeddings = read_embeddings(mirna_csv_path)
disease_embeddings = read_embeddings(disease_csv_path)


# Read relationships from the merged data CSV file
relationships_to_include = []
with open(relation_csv_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        mirna_name = row['miRNA']
        disease_name = row['disease']
        relationships_to_include.append((mirna_name, disease_name))

# Create the JSON structure
relationships = []
for mirna_name, disease_name in relationships_to_include:
    if mirna_name in mirna_embeddings and disease_name in disease_embeddings:
        relationship = {
            "miRNA": {
                "properties": {
                    "name": mirna_name,
                    "embedding": mirna_embeddings[mirna_name]
                }
            },
            "relation": {
                "type": "CONNECTED"
            },
            "disease": {
                "properties": {
                    "name": disease_name,
                    "embedding": disease_embeddings[disease_name]
                }
            }
        }
        relationships.append(relationship)

# Save to JSON file
with open(output_json_path, 'w') as json_file:
    json.dump(relationships, json_file, indent=2)

print(f"JSON file saved to {output_json_path}")
