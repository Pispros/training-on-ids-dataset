import pandas as pd
import os
import math
import json
import csv

sourceDirs = ['./CIC-IDS2017/converted_to_CICIDS2018', './CIC-IDS2018']
destinationDirs = ['./CIC-IDS2017.reformed', './CIC-IDS2018.reformed']
max_size_mb = 460  # Max size in MB

def extract_data(sourcePath, destinationDir):
    # Charger les données DataFrame
    df = pd.read_csv(sourcePath, low_memory=False)
    trainDataLength = math.floor(len(df) * 0.8 + 1)
    validateDataEndIndex = math.floor(len(df) * 0.9)
    file_size = os.path.getsize(sourcePath) / 1048576 # 1024 x 1024
    print(f"Source: {sourcePath}, File size: {file_size:.2f} MB")
    print(f"length {len(df)}")
    print(f"trainDataLength {trainDataLength}")
    print(f"validateDataEndIndex {validateDataEndIndex}")

    with open(sourcePath, 'r') as file:
        lines = file.readlines()

    data = lines[1:] 

    train_data = data[:trainDataLength]
    validate_data = data[trainDataLength:validateDataEndIndex]
    test_data = data[validateDataEndIndex:]

    with open(f"{destinationDir}/train.csv", 'a') as file:
        file.writelines(train_data)
    
    with open(f"{destinationDir}/validate.csv", 'a') as file:
        file.writelines(validate_data)

    with open(f"{destinationDir}/test.csv", 'a') as file:
        file.writelines(test_data)

def main():
    # Récupérer les fichiers de Dataset dans les dossiers source et les scinder en train/validate/test.csv en suivant le ratio 80-10-10

    custom_header = ("Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,"
                     "Fwd Pkt Len Max,Fwd Pkt Len Min,Fwd Pkt Len Mean,Fwd Pkt Len Std,Bwd Pkt Len Max,Bwd Pkt Len Min,"
                     "Bwd Pkt Len Mean,Bwd Pkt Len Std,Flow Byts/s,Flow Pkts/s,Flow IAT Mean,Flow IAT Std,Flow IAT Max,"
                     "Flow IAT Min,Fwd IAT Tot,Fwd IAT Mean,Fwd IAT Std,Fwd IAT Max,Fwd IAT Min,Bwd IAT Tot,Bwd IAT Mean,"
                     "Bwd IAT Std,Bwd IAT Max,Bwd IAT Min,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,Fwd Header Len,"
                     "Bwd Header Len,Fwd Pkts/s,Bwd Pkts/s,Pkt Len Min,Pkt Len Max,Pkt Len Mean,Pkt Len Std,Pkt Len Var,"
                     "FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,URG Flag Cnt,CWE Flag Count,ECE Flag Cnt,"
                     "Down/Up Ratio,Pkt Size Avg,Fwd Seg Size Avg,Bwd Seg Size Avg,Fwd Byts/b Avg,Fwd Pkts/b Avg,Fwd Blk Rate Avg,"
                     "Bwd Byts/b Avg,Bwd Pkts/b Avg,Bwd Blk Rate Avg,Subflow Fwd Pkts,Subflow Fwd Byts,Subflow Bwd Pkts,Subflow Bwd Byts,"
                     "Init Fwd Win Byts,Init Bwd Win Byts,Fwd Act Data Pkts,Fwd Seg Size Min,Active Mean,Active Std,Active Max,"
                     "Active Min,Idle Mean,Idle Std,Idle Max,Idle Min,Label\n")


    dirIndex = 0
    for sourceDir in sourceDirs:
        if not os.path.exists(destinationDirs[dirIndex]):
            os.makedirs(destinationDirs[dirIndex])

        if not os.path.exists(f"{destinationDirs[dirIndex]}/train.csv"):
           with open(f"{destinationDirs[dirIndex]}/train.csv", 'w') as file:
                file.write(custom_header)

        if not os.path.exists(f"{destinationDirs[dirIndex]}/train.json"):
           open(f"{destinationDirs[dirIndex]}/train.json", 'w')

        if not os.path.exists(f"{destinationDirs[dirIndex]}/validate.csv"):
            with open(f"{destinationDirs[dirIndex]}/validate.csv", 'w') as file:
                file.write(custom_header)
        
        if not os.path.exists(f"{destinationDirs[dirIndex]}/validate.json"):
           open(f"{destinationDirs[dirIndex]}/validate.json", 'w')

        if not os.path.exists(f"{destinationDirs[dirIndex]}/test.csv",):
            with open(f"{destinationDirs[dirIndex]}/test.csv", 'w') as file:
                file.write(custom_header)

        if not os.path.exists(f"{destinationDirs[dirIndex]}/test.json"):
           open(f"{destinationDirs[dirIndex]}/test.json", 'w')

        f = []
        for (dirpath, dirnames, filenames) in os.walk(sourceDir):
            f.extend(filenames)
            break
        for filePath in f:
            if '.csv' in filePath:
                extract_data(f"{sourceDirs[dirIndex]}/{filePath}", destinationDirs[dirIndex])
        
        convert_csv_to_json(f"{destinationDirs[dirIndex]}/train.csv", f"{destinationDirs[dirIndex]}/train.json")
        convert_csv_to_json(f"{destinationDirs[dirIndex]}/validate.csv", f"{destinationDirs[dirIndex]}/validate.json")
        convert_csv_to_json(f"{destinationDirs[dirIndex]}/test.csv", f"{destinationDirs[dirIndex]}/test.json")

        split_json_by_size(f"{destinationDirs[dirIndex]}/train.json", destinationDirs[dirIndex] + "/train_{}.json")
        split_json_by_size(f"{destinationDirs[dirIndex]}/validate.json", destinationDirs[dirIndex] + "/validate_{}.json")
        split_json_by_size(f"{destinationDirs[dirIndex]}/test.json", destinationDirs[dirIndex] + "/test_{}.json")

        dirIndex+=1
        

def convert_csv_to_json(input_csv, output_json):
    # Liste pour stocker les données JSON
    data = []

    # Lire le fichier CSV
    with open(input_csv, mode='r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        # Parcourir chaque ligne du fichier CSV
        for row in csv_reader:
            # Construire le prompt à partir des attributs
            prompt = ', '.join(f"{key}: {value}" for key, value in row.items() if key != 'Label')

            # Construire la completion à partir du label
            completion = row['Label']

            # Ajouter l'objet JSON à la liste
            data.append({
                "prompt": prompt,
                "completion": completion
            })

    # Écrire les données JSON dans un fichier
    with open(output_json, mode='w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Le fichier JSON a été créé avec succès à {output_json}")

def split_json_by_size(input_file, output_file_template):
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
    current_chunk = []
    current_size = 0
    chunk_index = 0

    # Open the input file and load the data
    with open(input_file, 'r') as f:
        data = json.load(f)

    for item in data:
        # Calculate the size of the current item when serialized
        item_size = len(json.dumps(item).encode('utf-8'))

        # If adding the item exceeds the max size, save the current chunk and start a new one
        if current_size + item_size > max_size_bytes:
            output_file = output_file_template.format(chunk_index)
            with open(output_file, 'w') as out_f:
                json.dump(current_chunk, out_f)
            
            chunk_index += 1
            current_chunk = []
            current_size = 0

        # Add the current item to the chunk
        current_chunk.append(item)
        current_size += item_size

    # Save the final chunk
    if current_chunk:
        output_file = output_file_template.format(chunk_index)
        with open(output_file, 'w') as out_f:
            json.dump(current_chunk, out_f)

main()
