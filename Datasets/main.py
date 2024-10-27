import pandas as pd
import os
import math

sourceDirs = ['./CIC-IDS2017/converted_to_CICIDS2018', './CIC-IDS2018']
destinationDirs = ['./CIC-IDS2017.reformed', './CIC-IDS2018.reformed']

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

    custom_header = ("Dst Port\tProtocol\tTimestamp\tFlow Duration\tTot Fwd Pkts\tTot Bwd Pkts\tTotLen Fwd Pkts\tTotLen Bwd Pkts\t"
                     "Fwd Pkt Len Max\tFwd Pkt Len Min\tFwd Pkt Len Mean\tFwd Pkt Len Std\tBwd Pkt Len Max\tBwd Pkt Len Min\t"
                     "Bwd Pkt Len Mean\tBwd Pkt Len Std\tFlow Byts/s\tFlow Pkts/s\tFlow IAT Mean\tFlow IAT Std\tFlow IAT Max\t"
                     "Flow IAT Min\tFwd IAT Tot\tFwd IAT Mean\tFwd IAT Std\tFwd IAT Max\tFwd IAT Min\tBwd IAT Tot\tBwd IAT Mean\t"
                     "Bwd IAT Std\tBwd IAT Max\tBwd IAT Min\tFwd PSH Flags\tBwd PSH Flags\tFwd URG Flags\tBwd URG Flags\tFwd Header Len\t"
                     "Bwd Header Len\tFwd Pkts/s\tBwd Pkts/s\tPkt Len Min\tPkt Len Max\tPkt Len Mean\tPkt Len Std\tPkt Len Var\t"
                     "FIN Flag Cnt\tSYN Flag Cnt\tRST Flag Cnt\tPSH Flag Cnt\tACK Flag Cnt\tURG Flag Cnt\tCWE Flag Count\tECE Flag Cnt\t"
                     "Down/Up Ratio\tPkt Size Avg\tFwd Seg Size Avg\tBwd Seg Size Avg\tFwd Byts/b Avg\tFwd Pkts/b Avg\tFwd Blk Rate Avg\t"
                     "Bwd Byts/b Avg\tBwd Pkts/b Avg\tBwd Blk Rate Avg\tSubflow Fwd Pkts\tSubflow Fwd Byts\tSubflow Bwd Pkts\tSubflow Bwd Byts\t"
                     "Init Fwd Win Byts\tInit Bwd Win Byts\tFwd Act Data Pkts\tFwd Seg Size Min\tActive Mean\tActive Std\tActive Max\t"
                     "Active Min\tIdle Mean\tIdle Std\tIdle Max\tIdle Min\tLabel\n")


    dirIndex = 0
    for sourceDir in sourceDirs:
        if not os.path.exists(destinationDirs[dirIndex]):
            os.makedirs(destinationDirs[dirIndex])

        if not os.path.exists(f"{destinationDirs[dirIndex]}/train.csv"):
           with open(f"{destinationDirs[dirIndex]}/train.csv", 'w') as file:
                file.write(custom_header)

        if not os.path.exists(f"{destinationDirs[dirIndex]}/validate.csv"):
            with open(f"{destinationDirs[dirIndex]}/validate.csv", 'w') as file:
                file.write(custom_header)

        if not os.path.exists(f"{destinationDirs[dirIndex]}/test.csv",):
            with open(f"{destinationDirs[dirIndex]}/test.csv", 'w') as file:
                file.write(custom_header)

        f = []
        for (dirpath, dirnames, filenames) in os.walk(sourceDir):
            f.extend(filenames)
            break
        for filePath in f:
            if '.csv' in filePath:
                extract_data(f"{sourceDirs[dirIndex]}/{filePath}", destinationDirs[dirIndex])
        dirIndex+=1
    
main()
