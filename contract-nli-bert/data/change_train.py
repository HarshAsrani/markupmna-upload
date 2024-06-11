import json

# Path to the source JSON file
input_file_path = 'train.json'
output_file_path = 'train_1.json'

# Read the JSON data from the file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Extract the first 16 documents (assuming the key for documents is 'documents')
# data['documents'] = data['documents'][:1]  # Keep only the first 16 documents
for document in data['documents']:
    document['url'] = 'something'
# The labels are assumed to be intact as per the requirement
# Write the modified data to a new JSON file
with open(output_file_path, 'w') as file:
    json.dump(data, file, indent=4)

