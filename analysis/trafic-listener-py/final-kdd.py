from scapy.all import sniff, TCP, UDP, ICMP, IP
import time
import csv

# Dictionnaires pour stocker les connexions et les statistiques
connections = {}
recent_connections = []
dst_host_stats = {}

def get_service(port, protocol):
    services = {
        20: 'ftp_data',
        21: 'ftp',
        22: 'ssh',
        23: 'telnet',
        25: 'smtp',
        53: 'domain',
        80: 'http',
        110: 'pop_3',
        111: 'sunrpc',
        143: 'imap4',
        443: 'https',
        # Ajoutez d'autres services si nécessaire
    }
    return services.get(port, 'other')

def get_connection_state(flags):
    if 'S' in flags and 'A' in flags:
        return 'S1'
    elif 'S' in flags:
        return 'S0'
    elif 'R' in flags:
        return 'REJ'
    elif 'F' in flags and 'A' in flags:
        return 'SF'
    else:
        return 'OTH'

def packet_callback(packet):
    current_time = time.time()

    if IP in packet:
        ip_layer = packet[IP]
        protocol = ip_layer.proto
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        # Déterminer le type de protocole
        if TCP in packet:
            proto = 'tcp'
            transport_layer = packet[TCP]
        elif UDP in packet:
            proto = 'udp'
            transport_layer = packet[UDP]
        elif ICMP in packet:
            proto = 'icmp'
            transport_layer = packet[ICMP]
        else:
            proto = 'other'
            transport_layer = None

        if transport_layer:
            src_port = transport_layer.sport if hasattr(transport_layer, 'sport') else 0
            dst_port = transport_layer.dport if hasattr(transport_layer, 'dport') else 0
            flags = str(transport_layer.flags) if hasattr(transport_layer, 'flags') else ''
        else:
            src_port = 0
            dst_port = 0
            flags = ''

        # Générer une clé unique pour la connexion
        conn_key = (src_ip, src_port, dst_ip, dst_port, proto)

        if conn_key not in connections:
            # Nouvelle connexion
            connections[conn_key] = {
                'start_time': current_time,
                'src_bytes': 0,
                'dst_bytes': 0,
                'src_packets': 0,
                'dst_packets': 0,
                'protocol_type': proto,
                'service': get_service(dst_port, proto),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'flags': [],
                'land': int(src_ip == dst_ip and src_port == dst_port),
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_hot_login': 0,
                'is_guest_login': 0,
                'count': 0,
                'srv_count': 0,
                'serror_rate': 0.0,
                'srv_serror_rate': 0.0,
                'rerror_rate': 0.0,
                'srv_rerror_rate': 0.0,
                'same_srv_rate': 0.0,
                'diff_srv_rate': 0.0,
                'srv_diff_host_rate': 0.0,
                # Caractéristiques basées sur l'hôte cible
                'dst_host_count': 0,
                'dst_host_srv_count': 0,
                'dst_host_same_srv_rate': 0.0,
                'dst_host_diff_srv_rate': 0.0,
                'dst_host_same_src_port_rate': 0.0,
                'dst_host_srv_diff_host_rate': 0.0,
                'dst_host_serror_rate': 0.0,
                'dst_host_srv_serror_rate': 0.0,
                'dst_host_rerror_rate': 0.0,
                'dst_host_srv_rerror_rate': 0.0
            }

        conn = connections[conn_key]

        # Mettre à jour les compteurs
        payload_size = len(transport_layer.payload) if hasattr(transport_layer, 'payload') else 0
        if src_ip == conn['src_ip']:
            conn['src_bytes'] += payload_size
            conn['src_packets'] += 1
        else:
            conn['dst_bytes'] += payload_size
            conn['dst_packets'] += 1

        # Enregistrer les flags
        if flags:
            conn['flags'].append(flags)

        # Vérifier les paquets urgents
        if 'U' in flags:
            conn['urgent'] += 1

        # Vérifier les fragments incorrects (simplifié)
        if ip_layer.flags == 1 or ip_layer.frag > 0:
            conn['wrong_fragment'] += 1

        # Mettre à jour les caractéristiques temporelles
        update_temporal_features(conn, current_time)

        # Mettre à jour les caractéristiques basées sur l'hôte cible
        update_dst_host_features(conn, current_time)

        # Détecter la fin de la connexion
        if proto == 'tcp' and ('F' in flags or 'R' in flags):
            # Calculer la durée
            conn['end_time'] = current_time
            conn['duration'] = conn['end_time'] - conn['start_time']

            # Déterminer l'état de la connexion
            conn['flag'] = get_connection_state(''.join(conn['flags']))

            # Enregistrer la connexion
            save_connection(conn)

            # Supprimer la connexion du dictionnaire
            del connections[conn_key]

def update_temporal_features(conn, current_time):
    # Ajouter la connexion à la liste des connexions récentes
    recent_connections.append({
        'time': current_time,
        'src_ip': conn['src_ip'],
        'dst_ip': conn['dst_ip'],
        'service': conn['service'],
        'status': conn['flag'] if 'flag' in conn else 'OTH'
    })

    # Supprimer les connexions plus anciennes que 2 secondes
    window = 2  # secondes
    recent_connections[:] = [c for c in recent_connections if current_time - c['time'] <= window]

    # Calculer les caractéristiques temporelles
    same_host_conns = [c for c in recent_connections if c['dst_ip'] == conn['dst_ip']]
    same_service_conns = [c for c in recent_connections if c['service'] == conn['service']]

    conn['count'] = len(same_host_conns)
    conn['srv_count'] = len(same_service_conns)

    # Calcul des taux d'erreur
    serror_conns = [c for c in same_host_conns if c['status'] in ['S0', 'S1', 'S2', 'S3']]
    conn['serror_rate'] = len(serror_conns) / conn['count'] if conn['count'] > 0 else 0.0

    srv_serror_conns = [c for c in same_service_conns if c['status'] in ['S0', 'S1', 'S2', 'S3']]
    conn['srv_serror_rate'] = len(srv_serror_conns) / conn['srv_count'] if conn['srv_count'] > 0 else 0.0

    rerror_conns = [c for c in same_host_conns if c['status'] == 'REJ']
    conn['rerror_rate'] = len(rerror_conns) / conn['count'] if conn['count'] > 0 else 0.0

    srv_rerror_conns = [c for c in same_service_conns if c['status'] == 'REJ']
    conn['srv_rerror_rate'] = len(srv_rerror_conns) / conn['srv_count'] if conn['srv_count'] > 0 else 0.0

    # Taux de services identiques et différents
    conn['same_srv_rate'] = conn['srv_count'] / conn['count'] if conn['count'] > 0 else 0.0
    conn['diff_srv_rate'] = 1.0 - conn['same_srv_rate']

    # Taux de connexions vers des hôtes différents pour le même service
    srv_diff_host_conns = set([c['dst_ip'] for c in same_service_conns if c['dst_ip'] != conn['dst_ip']])
    conn['srv_diff_host_rate'] = len(srv_diff_host_conns) / conn['srv_count'] if conn['srv_count'] > 0 else 0.0

def update_dst_host_features(conn, current_time):
    dst_ip = conn['dst_ip']
    if dst_ip not in dst_host_stats:
        dst_host_stats[dst_ip] = []

    dst_host_stats[dst_ip].append({
        'time': current_time,
        'service': conn['service'],
        'src_port': conn['src_port'],
        'src_ip': conn['src_ip'],
        'flag': conn.get('flag', 'OTH')
    })

    # Garder seulement les 100 dernières connexions
    if len(dst_host_stats[dst_ip]) > 100:
        dst_host_stats[dst_ip] = dst_host_stats[dst_ip][-100:]

    # Calculer les caractéristiques
    conn['dst_host_count'] = len(dst_host_stats[dst_ip])

    # dst_host_srv_count
    same_srv_conns = [c for c in dst_host_stats[dst_ip] if c['service'] == conn['service']]
    conn['dst_host_srv_count'] = len(same_srv_conns)

    # dst_host_same_srv_rate
    conn['dst_host_same_srv_rate'] = conn['dst_host_srv_count'] / conn['dst_host_count'] if conn['dst_host_count'] > 0 else 0.0

    # dst_host_diff_srv_rate
    diff_srv_conns = conn['dst_host_count'] - conn['dst_host_srv_count']
    conn['dst_host_diff_srv_rate'] = diff_srv_conns / conn['dst_host_count'] if conn['dst_host_count'] > 0 else 0.0

    # dst_host_same_src_port_rate
    same_src_port_conns = [c for c in dst_host_stats[dst_ip] if c['src_port'] == conn['src_port']]
    conn['dst_host_same_src_port_rate'] = len(same_src_port_conns) / conn['dst_host_count'] if conn['dst_host_count'] > 0 else 0.0

    # dst_host_srv_diff_host_rate
    srv_diff_host_conns = set(c['src_ip'] for c in same_srv_conns if c['src_ip'] != conn['src_ip'])
    conn['dst_host_srv_diff_host_rate'] = len(srv_diff_host_conns) / conn['dst_host_srv_count'] if conn['dst_host_srv_count'] > 0 else 0.0

    # dst_host_serror_rate
    serror_conns = [c for c in dst_host_stats[dst_ip] if c['flag'] in ['S0', 'S1', 'S2', 'S3']]
    conn['dst_host_serror_rate'] = len(serror_conns) / conn['dst_host_count'] if conn['dst_host_count'] > 0 else 0.0

    # dst_host_srv_serror_rate
    serror_srv_conns = [c for c in same_srv_conns if c['flag'] in ['S0', 'S1', 'S2', 'S3']]
    conn['dst_host_srv_serror_rate'] = len(serror_srv_conns) / conn['dst_host_srv_count'] if conn['dst_host_srv_count'] > 0 else 0.0

    # dst_host_rerror_rate
    rerror_conns = [c for c in dst_host_stats[dst_ip] if c['flag'] == 'REJ']
    conn['dst_host_rerror_rate'] = len(rerror_conns) / conn['dst_host_count'] if conn['dst_host_count'] > 0 else 0.0

    # dst_host_srv_rerror_rate
    rerror_srv_conns = [c for c in same_srv_conns if c['flag'] == 'REJ']
    conn['dst_host_srv_rerror_rate'] = len(rerror_srv_conns) / conn['dst_host_srv_count'] if conn['dst_host_srv_count'] > 0 else 0.0

def save_connection(conn):
    fieldnames = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_hot_login', 'is_guest_login',
        'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    with open('connections.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Écrire l'en-tête si le fichier est vide
        if csvfile.tell() == 0:
            writer.writeheader()

        record = {key: conn.get(key, 0) for key in fieldnames}
        writer.writerow(record)

    print(f"Connexion enregistrée : {conn}")

# Démarrer la capture des paquets
sniff(iface="enx144fd7c5e7ea", prn=packet_callback, store=0)
