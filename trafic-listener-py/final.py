import time
import csv
from scapy.all import sniff, IP, TCP, UDP
import numpy as np

# Dictionnaire pour stocker les flux actifs
flows = {}

# Liste des caractéristiques à extraire (colonnes)
feature_names = [
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

# Fonction pour générer un identifiant de flux unique
def generate_flow_id(pkt):
    ip_layer = pkt[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    protocol = ip_layer.proto

    if TCP in pkt or UDP in pkt:
        transport_layer = pkt[TCP] if TCP in pkt else pkt[UDP]
        src_port = transport_layer.sport
        dst_port = transport_layer.dport
    else:
        src_port = 0
        dst_port = 0

    # Flow ID unique basé sur les adresses IP, les ports et le protocole
    flow_id = f"{src_ip}-{src_port}-{dst_ip}-{dst_port}-{protocol}"
    rev_flow_id = f"{dst_ip}-{dst_port}-{src_ip}-{src_port}-{protocol}"
    return flow_id, rev_flow_id

# Fonction de rappel pour chaque paquet capturé
def packet_callback(pkt):
    if IP in pkt:
        current_time = time.time() * 1000000  # Temps en microsecondes
        flow_id, rev_flow_id = generate_flow_id(pkt)
        is_reverse = False

        if flow_id in flows:
            flow = flows[flow_id]
        elif rev_flow_id in flows:
            flow = flows[rev_flow_id]
            is_reverse = True
        else:
            # Nouveau flux
            print(f"Nouveau flux : {flow_id}")
            flow = {
                'Src IP': pkt[IP].src,
                'Src Port': pkt[TCP].sport if TCP in pkt else (pkt[UDP].sport if UDP in pkt else 0),
                'Dst IP': pkt[IP].dst,
                'Dst Port': pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
                'Protocol': pkt[IP].proto,
                'Timestamp': current_time,
                'Flow Duration': 0,
                'Tot Fwd Pkts': 0,
                'Tot Bwd Pkts': 0,
                'TotLen Fwd Pkts': 0,
                'TotLen Bwd Pkts': 0,
                'Fwd Pkt Len Max': 0,
                'Fwd Pkt Len Min': float('inf'),
                'Fwd Pkt Len Mean': 0,
                'Fwd Pkt Len Std': 0,
                'Bwd Pkt Len Max': 0,
                'Bwd Pkt Len Min': float('inf'),
                'Bwd Pkt Len Mean': 0,
                'Bwd Pkt Len Std': 0,
                'Flow Byts/s': 0,
                'Flow Pkts/s': 0,
                'Flow IAT Mean': 0,
                'Flow IAT Std': 0,
                'Flow IAT Max': 0,
                'Flow IAT Min': 0,
                'Fwd IAT Tot': 0,
                'Fwd IAT Mean': 0,
                'Fwd IAT Std': 0,
                'Fwd IAT Max': 0,
                'Fwd IAT Min': 0,
                'Bwd IAT Tot': 0,
                'Bwd IAT Mean': 0,
                'Bwd IAT Std': 0,
                'Bwd IAT Max': 0,
                'Bwd IAT Min': 0,
                'Fwd PSH Flags': 0,
                'Bwd PSH Flags': 0,
                'Fwd URG Flags': 0,
                'Bwd URG Flags': 0,
                'Fwd Header Len': 0,
                'Bwd Header Len': 0,
                'Fwd Pkts/s': 0,
                'Bwd Pkts/s': 0,
                'Pkt Len Min': float('inf'),
                'Pkt Len Max': 0,
                'Pkt Len Mean': 0,
                'Pkt Len Std': 0,
                'Pkt Len Var': 0,
                'FIN Flag Cnt': 0,
                'SYN Flag Cnt': 0,
                'RST Flag Cnt': 0,
                'PSH Flag Cnt': 0,
                'ACK Flag Cnt': 0,
                'URG Flag Cnt': 0,
                'CWE Flag Count': 0,
                'ECE Flag Cnt': 0,
                'Down/Up Ratio': 0,
                'Pkt Size Avg': 0,
                'Fwd Seg Size Avg': 0,
                'Bwd Seg Size Avg': 0,
                'Fwd Byts/b Avg': np.nan,
                'Fwd Pkts/b Avg': np.nan,
                'Fwd Blk Rate Avg': np.nan,
                'Bwd Byts/b Avg': np.nan,
                'Bwd Pkts/b Avg': np.nan,
                'Bwd Blk Rate Avg': np.nan,
                'Subflow Fwd Pkts': 0,
                'Subflow Fwd Byts': 0,
                'Subflow Bwd Pkts': 0,
                'Subflow Bwd Byts': 0,
                'Init Fwd Win Byts': pkt[TCP].window if TCP in pkt else 0,
                'Init Bwd Win Byts': 0,
                'Fwd Act Data Pkts': 0,
                'Fwd Seg Size Min': float('inf'),
                'Active Mean': 0,
                'Active Std': 0,
                'Active Max': 0,
                'Active Min': 0,
                'Idle Mean': 0,
                'Idle Std': 0,
                'Idle Max': 0,
                'Idle Min': 0,
                'Label': 'BENIGN',  # Par défaut, vous pouvez changer cela selon vos besoins
                # Variables internes pour calcul
                'Flow ID': flow_id,
                'Last Packet Time': current_time,
                'Flow IATs': [],
                'Fwd IATs': [],
                'Bwd IATs': [],
                'Fwd Packet Times': [],
                'Bwd Packet Times': [],
                'Fwd Packet Lengths': [],
                'Bwd Packet Lengths': [],
                'Packet Lengths': [],
                'Flag Counts': {
                    'FIN': 0,
                    'SYN': 0,
                    'RST': 0,
                    'PSH': 0,
                    'ACK': 0,
                    'URG': 0,
                    'CWE': 0,
                    'ECE': 0
                }
            }
            flows[flow_id] = flow

        # Mettre à jour le flux avec les informations du paquet
        update_flow(flow, pkt, current_time, is_reverse)

def update_flow(flow, pkt, current_time, is_reverse):
    # Mettre à jour la durée du flux
    flow['Flow Duration'] = current_time - flow['Timestamp']

    # Inter Arrival Time (IAT)
    flow_iat = current_time - flow['Last Packet Time']
    flow['Flow IATs'].append(flow_iat)
    flow['Last Packet Time'] = current_time

    # Longueur du paquet
    pkt_length = len(pkt)
    flow['Packet Lengths'].append(pkt_length)
    flow['Pkt Len Min'] = min(flow['Pkt Len Min'], pkt_length)
    flow['Pkt Len Max'] = max(flow['Pkt Len Max'], pkt_length)

    # Gérer les paquets avant et arrière
    if not is_reverse:
        # Paquet avant (forward)
        flow['Tot Fwd Pkts'] += 1
        flow['TotLen Fwd Pkts'] += pkt_length
        flow['Fwd Packet Lengths'].append(pkt_length)
        flow['Fwd Pkt Len Min'] = min(flow['Fwd Pkt Len Min'], pkt_length)
        flow['Fwd Pkt Len Max'] = max(flow['Fwd Pkt Len Max'], pkt_length)
        flow['Subflow Fwd Pkts'] += 1
        flow['Subflow Fwd Byts'] += pkt_length
        flow['Fwd Packet Times'].append(current_time)
        flow['Fwd Seg Size Min'] = min(flow['Fwd Seg Size Min'], pkt_length)

        # Fwd IAT
        if len(flow['Fwd Packet Times']) > 1:
            fwd_iat = current_time - flow['Fwd Packet Times'][-2]
            flow['Fwd IATs'].append(fwd_iat)

        # Flags TCP
        if TCP in pkt:
            tcp_layer = pkt[TCP]
            flags = tcp_layer.flags

            # Mettre à jour les flags
            flow_ended = update_flags(flow, flags, is_reverse)
            if flow_ended:
                return

            # Header Length
            hdr_len = tcp_layer.dataofs * 4
            flow['Fwd Header Len'] += hdr_len

            # Initial Window Size
            if flow['Init Fwd Win Byts'] == 0:
                flow['Init Fwd Win Byts'] = tcp_layer.window

    else:
        # Paquet arrière (backward)
        flow['Tot Bwd Pkts'] += 1
        flow['TotLen Bwd Pkts'] += pkt_length
        flow['Bwd Packet Lengths'].append(pkt_length)
        flow['Bwd Pkt Len Min'] = min(flow['Bwd Pkt Len Min'], pkt_length)
        flow['Bwd Pkt Len Max'] = max(flow['Bwd Pkt Len Max'], pkt_length)
        flow['Subflow Bwd Pkts'] += 1
        flow['Subflow Bwd Byts'] += pkt_length
        flow['Bwd Packet Times'].append(current_time)

        # Bwd IAT
        if len(flow['Bwd Packet Times']) > 1:
            bwd_iat = current_time - flow['Bwd Packet Times'][-2]
            flow['Bwd IATs'].append(bwd_iat)

        # Flags TCP
        if TCP in pkt:
            tcp_layer = pkt[TCP]
            flags = tcp_layer.flags

            # Mettre à jour les flags
            flow_ended = update_flags(flow, flags, is_reverse)
            if flow_ended:
                return

            # Header Length
            hdr_len = tcp_layer.dataofs * 4
            flow['Bwd Header Len'] += hdr_len

            # Initial Window Size
            if flow['Init Bwd Win Byts'] == 0:
                flow['Init Bwd Win Byts'] = tcp_layer.window

    # Vérifier la fin du flux par timeout
    if flow_timeout(flow, current_time):
        print(f"Fin du flux par timeout : {flow['Flow ID']}")
        # Calculer les caractéristiques finales et enregistrer le flux
        calculate_features_and_save(flow)
        # Supprimer le flux des flux actifs
        del flows[flow['Flow ID']]

def update_flags(flow, flags, is_reverse):
    # Flags TCP
    flow_ended = False
    if flags & 0x01:
        flow['Flag Counts']['FIN'] += 1
        flow['FIN Flag Cnt'] += 1
        # Détection de la fin du flux par flag FIN
        print(f"Fin du flux détectée par flag FIN : {flow['Flow ID']}")
        calculate_features_and_save(flow)
        del flows[flow['Flow ID']]
        flow_ended = True
    if flags & 0x02:
        flow['Flag Counts']['SYN'] += 1
        flow['SYN Flag Cnt'] += 1
    if flags & 0x04:
        flow['Flag Counts']['RST'] += 1
        flow['RST Flag Cnt'] += 1
        # Détection de la fin du flux par flag RST
        print(f"Fin du flux détectée par flag RST : {flow['Flow ID']}")
        calculate_features_and_save(flow)
        del flows[flow['Flow ID']]
        flow_ended = True
    if flags & 0x08:
        flow['Flag Counts']['PSH'] += 1
        flow['PSH Flag Cnt'] += 1
        if not is_reverse:
            flow['Fwd PSH Flags'] += 1
        else:
            flow['Bwd PSH Flags'] += 1
    if flags & 0x10:
        flow['Flag Counts']['ACK'] += 1
        flow['ACK Flag Cnt'] += 1
    if flags & 0x20:
        flow['Flag Counts']['URG'] += 1
        flow['URG Flag Cnt'] += 1
        if not is_reverse:
            flow['Fwd URG Flags'] += 1
        else:
            flow['Bwd URG Flags'] += 1
    if flags & 0x40:
        flow['Flag Counts']['ECE'] += 1
        flow['ECE Flag Cnt'] += 1
    if flags & 0x80:
        flow['Flag Counts']['CWE'] += 1
        flow['CWE Flag Count'] += 1
    return flow_ended

def flow_timeout(flow, current_time):
    # Définir un timeout pour les flux inactifs (en microsecondes)
    inactive_timeout = 10000000  # 10 secondes
    return (current_time - flow['Last Packet Time']) > inactive_timeout

def calculate_features_and_save(flow):
    # Calcul des caractéristiques supplémentaires
    flow_duration = flow['Flow Duration'] / 1000000  # Convertir en secondes
    total_packets = flow['Tot Fwd Pkts'] + flow['Tot Bwd Pkts']
    total_bytes = flow['TotLen Fwd Pkts'] + flow['TotLen Bwd Pkts']

    # Flow Bytes/s et Flow Packets/s
    if flow_duration > 0:
        flow['Flow Byts/s'] = total_bytes / flow_duration
        flow['Flow Pkts/s'] = total_packets / flow_duration
        flow['Fwd Pkts/s'] = flow['Tot Fwd Pkts'] / flow_duration
        flow['Bwd Pkts/s'] = flow['Tot Bwd Pkts'] / flow_duration
    else:
        flow['Flow Byts/s'] = 0
        flow['Flow Pkts/s'] = 0
        flow['Fwd Pkts/s'] = 0
        flow['Bwd Pkts/s'] = 0

    # Calcul des statistiques des longueurs de paquets
    if flow['Packet Lengths']:
        flow['Pkt Len Mean'] = np.mean(flow['Packet Lengths'])
        flow['Pkt Len Std'] = np.std(flow['Packet Lengths'])
        flow['Pkt Len Var'] = np.var(flow['Packet Lengths'])
        flow['Pkt Size Avg'] = flow['Pkt Len Mean']
    else:
        flow['Pkt Len Mean'] = 0
        flow['Pkt Len Std'] = 0
        flow['Pkt Len Var'] = 0
        flow['Pkt Size Avg'] = 0

    # Calcul des statistiques pour les paquets forward
    if flow['Fwd Packet Lengths']:
        flow['Fwd Pkt Len Mean'] = np.mean(flow['Fwd Packet Lengths'])
        flow['Fwd Pkt Len Std'] = np.std(flow['Fwd Packet Lengths'])
        flow['Fwd Seg Size Avg'] = flow['Fwd Pkt Len Mean']
    else:
        flow['Fwd Pkt Len Mean'] = 0
        flow['Fwd Pkt Len Std'] = 0
        flow['Fwd Seg Size Avg'] = 0
        flow['Fwd Pkt Len Min'] = 0  # Si aucune valeur, on met à 0
        flow['Fwd Pkt Len Max'] = 0

    # Calcul des statistiques pour les paquets backward
    if flow['Bwd Packet Lengths']:
        flow['Bwd Pkt Len Mean'] = np.mean(flow['Bwd Packet Lengths'])
        flow['Bwd Pkt Len Std'] = np.std(flow['Bwd Packet Lengths'])
        flow['Bwd Seg Size Avg'] = flow['Bwd Pkt Len Mean']
    else:
        flow['Bwd Pkt Len Mean'] = 0
        flow['Bwd Pkt Len Std'] = 0
        flow['Bwd Seg Size Avg'] = 0
        flow['Bwd Pkt Len Min'] = 0  # Si aucune valeur, on met à 0
        flow['Bwd Pkt Len Max'] = 0

    # Calcul des IAT
    if flow['Flow IATs']:
        flow['Flow IAT Mean'] = np.mean(flow['Flow IATs'])
        flow['Flow IAT Std'] = np.std(flow['Flow IATs'])
        flow['Flow IAT Max'] = max(flow['Flow IATs'])
        flow['Flow IAT Min'] = min(flow['Flow IATs'])
    else:
        flow['Flow IAT Mean'] = 0
        flow['Flow IAT Std'] = 0
        flow['Flow IAT Max'] = 0
        flow['Flow IAT Min'] = 0

    if flow['Fwd IATs']:
        flow['Fwd IAT Tot'] = sum(flow['Fwd IATs'])
        flow['Fwd IAT Mean'] = np.mean(flow['Fwd IATs'])
        flow['Fwd IAT Std'] = np.std(flow['Fwd IATs'])
        flow['Fwd IAT Max'] = max(flow['Fwd IATs'])
        flow['Fwd IAT Min'] = min(flow['Fwd IATs'])
    else:
        flow['Fwd IAT Tot'] = 0
        flow['Fwd IAT Mean'] = 0
        flow['Fwd IAT Std'] = 0
        flow['Fwd IAT Max'] = 0
        flow['Fwd IAT Min'] = 0

    if flow['Bwd IATs']:
        flow['Bwd IAT Tot'] = sum(flow['Bwd IATs'])
        flow['Bwd IAT Mean'] = np.mean(flow['Bwd IATs'])
        flow['Bwd IAT Std'] = np.std(flow['Bwd IATs'])
        flow['Bwd IAT Max'] = max(flow['Bwd IATs'])
        flow['Bwd IAT Min'] = min(flow['Bwd IATs'])
    else:
        flow['Bwd IAT Tot'] = 0
        flow['Bwd IAT Mean'] = 0
        flow['Bwd IAT Std'] = 0
        flow['Bwd IAT Max'] = 0
        flow['Bwd IAT Min'] = 0

    # Down/Up Ratio
    if flow['Tot Bwd Pkts'] > 0:
        flow['Down/Up Ratio'] = flow['Tot Fwd Pkts'] / flow['Tot Bwd Pkts']
    else:
        flow['Down/Up Ratio'] = 0

    # Enregistrer le flux dans le fichier CSV
    save_flow_to_csv(flow)

def save_flow_to_csv(flow):
    output_file = 'flows.csv'

    # Vérifier si le fichier existe déjà
    file_exists = False
    try:
        with open(output_file, 'r'):
            file_exists = True
    except FileNotFoundError:
        pass

    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=feature_names)

        # Écrire l'en-tête si le fichier est nouveau
        if not file_exists:
            writer.writeheader()

        # Créer un dictionnaire avec les caractéristiques
        flow_record = {key: flow.get(key, 0) for key in feature_names}
        writer.writerow(flow_record)

    print(f"Flux enregistré : {flow['Flow ID']}")

def main():
    iface = 'enx144fd7c5e7ea'
    sniff(iface=iface, prn=packet_callback, store=0)

main()
