from scapy.all import sniff, TCP, IP

# Dictionnaire pour stocker les connexions
connections = {}

def packet_callback(packet):
    if TCP in packet and IP in packet:
        ip_layer = packet[IP]
        tcp_layer = packet[TCP]

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        src_port = tcp_layer.sport
        dst_port = tcp_layer.dport
        flags = tcp_layer.flags

        # Générer une clé unique pour la connexion
        conn_key = (src_ip, src_port, dst_ip, dst_port)

        if conn_key not in connections:
            # Nouvelle connexion
            connections[conn_key] = {
                'start_time': packet.time,
                'src_bytes': 0,
                'dst_bytes': 0,
                'packets': [],
                'flags': [],
                'protocol': 'tcp',
                'service': get_service(dst_port),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
            }

        conn = connections[conn_key]
        conn['packets'].append(packet)
        conn['flags'].append(flags)

        # Compter les octets
        payload_size = len(tcp_layer.payload)
        if src_ip == conn['src_ip']:
            conn['src_bytes'] += payload_size
        else:
            conn['dst_bytes'] += payload_size

        # Détecter la fin de la connexion
        if 'F' in flags:
            # Calculer la durée
            conn['end_time'] = packet.time
            conn['duration'] = conn['end_time'] - conn['start_time']

            # Enregistrer la connexion
            save_connection(conn)

            # Supprimer la connexion du dictionnaire
            del connections[conn_key]

def get_service(port):
    services = {80: 'http', 22: 'ssh', 21: 'ftp', 443: 'https'}
    return services.get(port, 'other')

def save_connection(conn):
    # Implémentez la logique pour sauvegarder la connexion
    print(f"Connexion terminée : {conn}")

sniff(iface="wlp0s20f3", prn=packet_callback, store=0)
