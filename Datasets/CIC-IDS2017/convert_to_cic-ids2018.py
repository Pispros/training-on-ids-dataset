import pandas as pd
import os

converted_to_CICIDS2018_dir = "converted_to_CICIDS2018"

def convertFile(filePath, convertedDirPath):
    # Chemin du fichier CICIDS2017
    input_file = filePath

    # Chemin du fichier de sortie CICIDS2018
    output_file = f"./{convertedDirPath}/{filePath.replace('.csv', '')}.converted.csv"

    # Charger les données du CICIDS2017 dans un DataFrame
    df_2017 = pd.read_csv(input_file)

    # Supprimer les espaces de début et de fin des noms de colonnes du CICIDS2017
    df_2017.columns = df_2017.columns.str.strip()

    # Liste des colonnes du CICIDS2018 que vous avez fournie
    columns_2018 = [
        'Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
        'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
        'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
        'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count',
        'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
        'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg',
        'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean',
        'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
    ]

    # Créer un dictionnaire de renommage pour mapper les colonnes du CICIDS2017 aux colonnes du CICIDS2018
    rename_dict = {
        # Colonnes existantes à mapper
        'Destination Port': 'Dst Port',
        'Flow Duration': 'Flow Duration',
        'Total Fwd Packets': 'Tot Fwd Pkts',
        'Total Backward Packets': 'Tot Bwd Pkts',
        'Total Length of Fwd Packets': 'TotLen Fwd Pkts',
        'Total Length of Bwd Packets': 'TotLen Bwd Pkts',
        'Fwd Packet Length Max': 'Fwd Pkt Len Max',
        'Fwd Packet Length Min': 'Fwd Pkt Len Min',
        'Fwd Packet Length Mean': 'Fwd Pkt Len Mean',
        'Fwd Packet Length Std': 'Fwd Pkt Len Std',
        'Bwd Packet Length Max': 'Bwd Pkt Len Max',
        'Bwd Packet Length Min': 'Bwd Pkt Len Min',
        'Bwd Packet Length Mean': 'Bwd Pkt Len Mean',
        'Bwd Packet Length Std': 'Bwd Pkt Len Std',
        'Flow Bytes/s': 'Flow Byts/s',
        'Flow Packets/s': 'Flow Pkts/s',
        'Flow IAT Mean': 'Flow IAT Mean',
        'Flow IAT Std': 'Flow IAT Std',
        'Flow IAT Max': 'Flow IAT Max',
        'Flow IAT Min': 'Flow IAT Min',
        'Fwd IAT Total': 'Fwd IAT Tot',
        'Fwd IAT Mean': 'Fwd IAT Mean',
        'Fwd IAT Std': 'Fwd IAT Std',
        'Fwd IAT Max': 'Fwd IAT Max',
        'Fwd IAT Min': 'Fwd IAT Min',
        'Bwd IAT Total': 'Bwd IAT Tot',
        'Bwd IAT Mean': 'Bwd IAT Mean',
        'Bwd IAT Std': 'Bwd IAT Std',
        'Bwd IAT Max': 'Bwd IAT Max',
        'Bwd IAT Min': 'Bwd IAT Min',
        'Fwd PSH Flags': 'Fwd PSH Flags',
        'Bwd PSH Flags': 'Bwd PSH Flags',
        'Fwd URG Flags': 'Fwd URG Flags',
        'Bwd URG Flags': 'Bwd URG Flags',
        'Fwd Header Length': 'Fwd Header Len',
        'Bwd Header Length': 'Bwd Header Len',
        'Fwd Packets/s': 'Fwd Pkts/s',
        'Bwd Packets/s': 'Bwd Pkts/s',
        'Min Packet Length': 'Pkt Len Min',
        'Max Packet Length': 'Pkt Len Max',
        'Packet Length Mean': 'Pkt Len Mean',
        'Packet Length Std': 'Pkt Len Std',
        'Packet Length Variance': 'Pkt Len Var',
        'FIN Flag Count': 'FIN Flag Cnt',
        'SYN Flag Count': 'SYN Flag Cnt',
        'RST Flag Count': 'RST Flag Cnt',
        'PSH Flag Count': 'PSH Flag Cnt',
        'ACK Flag Count': 'ACK Flag Cnt',
        'URG Flag Count': 'URG Flag Cnt',
        'CWE Flag Count': 'CWE Flag Count',
        'ECE Flag Count': 'ECE Flag Cnt',
        'Down/Up Ratio': 'Down/Up Ratio',
        'Average Packet Size': 'Pkt Size Avg',
        'Avg Fwd Segment Size': 'Fwd Seg Size Avg',
        'Avg Bwd Segment Size': 'Bwd Seg Size Avg',
        'Subflow Fwd Packets': 'Subflow Fwd Pkts',
        'Subflow Fwd Bytes': 'Subflow Fwd Byts',
        'Subflow Bwd Packets': 'Subflow Bwd Pkts',
        'Subflow Bwd Bytes': 'Subflow Bwd Byts',
        'Init_Win_bytes_forward': 'Init Fwd Win Byts',
        'Init_Win_bytes_backward': 'Init Bwd Win Byts',
        'act_data_pkt_fwd': 'Fwd Act Data Pkts',
        'min_seg_size_forward': 'Fwd Seg Size Min',
        'Active Mean': 'Active Mean',
        'Active Std': 'Active Std',
        'Active Max': 'Active Max',
        'Active Min': 'Active Min',
        'Idle Mean': 'Idle Mean',
        'Idle Std': 'Idle Std',
        'Idle Max': 'Idle Max',
        'Idle Min': 'Idle Min',
        'Label': 'Label'
    }

    # Renommer les colonnes du CICIDS2017
    df_2017.rename(columns=rename_dict, inplace=True)

    # Ajouter les colonnes manquantes du CICIDS2018 qui ne sont pas dans le CICIDS2017
    missing_columns = [col for col in columns_2018 if col not in df_2017.columns]
    for col in missing_columns:
        # Définir des valeurs par défaut appropriées pour chaque colonne
        if col == 'Protocol':
            df_2017[col] = 6  # Par exemple, 6 pour TCP
        elif col == 'Timestamp':
            df_2017[col] = pd.Timestamp.now()
        elif col == 'Src IP':
            df_2017[col] = '0.0.0.0'
        elif col == 'Dst IP':
            df_2017[col] = '0.0.0.0'
        elif col in ['Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg']:
            df_2017[col] = None  # Ces colonnes peuvent ne pas être présentes dans le CICIDS2017
        else:
            df_2017[col] = 0  # Valeur par défaut pour les autres colonnes numériques

    # Réorganiser les colonnes pour correspondre à l'ordre du CICIDS2018
    df_2017 = df_2017[columns_2018]

    # Exporter le DataFrame transformé en CSV
    df_2017.to_csv(output_file, index=False)

    print(f"Données transformées enregistrées sous {output_file}")


def main():
    f = []
    for (dirpath, dirnames, filenames) in os.walk('./'):
        f.extend(filenames)
        break
    
    if not os.path.exists(converted_to_CICIDS2018_dir):
        os.makedirs(converted_to_CICIDS2018_dir)
    
    for filePath in f:
        if '.csv' in filePath:
            convertFile(filePath, converted_to_CICIDS2018_dir)


main()



