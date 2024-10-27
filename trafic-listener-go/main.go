package main

import (
	"fmt"
	"log"

	"github.com/google/gopacket"
	"github.com/google/gopacket/pcap"
)

func main() {
	handle, err := pcap.OpenLive("wlp0s20f3", 1600, true, pcap.BlockForever)
	if err != nil {
		log.Fatal(err)
	}
	defer handle.Close()

	var filter string = "tcp port 22 or tcp port 80 or tcp port 443"
	err = handle.SetBPFFilter(filter)
	if err != nil {
		log.Fatal(err)
	}

	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
	fmt.Println("Écoute du trafic sur les ports 22, 80 et 443...")

	for packet := range packetSource.Packets() {
		fmt.Println("Paquet capturé : ", packet)
		// Vous pouvez extraire plus d'informations selon vos besoins
	}
}
