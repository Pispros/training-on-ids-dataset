import json
import os

dir_to_convert_files = "./CIC-IDS2018.reformed"

def convert_json_to_jsonl(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as json_file:
        data = json.load(json_file)

    # Check if the data is a list of JSON objects
    if isinstance(data, list):
        # Write each JSON object to a new line in the JSONL file
        with open(output_file, 'w') as jsonl_file:
            for item in data:
                jsonl_file.write(json.dumps(item) + '\n')
    else:
        print("The input JSON file does not contain a list of JSON objects.")

def main():
    f = []
    for (dirpath, dirnames, filenames) in os.walk(dir_to_convert_files):
        f.extend(filenames)
        break
    
    for filePath in f:
        if '.json' in filePath:
            convert_json_to_jsonl(f"{dir_to_convert_files}/{filePath}", f"{dir_to_convert_files}/{filePath.replace('.json', '.jsonl')}")

main()
