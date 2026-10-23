/*
 * ns-3 Network Simulation: AI-Powered Cybersecurity Posture Assessment
 * Topology: 3 Nodes (Node 0: Benign User, Node 1: Attacker, Node 2: Victim)
 * Links: Point-to-Point (10 Mbps DataRate, 2 ms Delay)
 * 
 * Traffic Profiles:
 *  - Benign Node: UDP 100 kbps, 512-byte packets, Duration: 0 - 20 s (continuous)
 *  - Attacker Node: UDP 5 Mbps, 1024-byte packets, Duration: 8 - 14 s (burst DoS)
 * 
 * Output: CSV trace file (ns3_packet_trace.csv) with timestamps, packet sizes, source, and label.
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include <fstream>
#include <iostream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("CybersecurityPostureNs3Sim");

std::ofstream g_traceFile;

void PacketRxTrace(std::string context, Ptr<const Packet> packet, const Address &address)
{
    double now = Simulator::Now().GetSeconds();
    uint32_t size = packet->GetSize();
    
    // Determine source traffic class based on simulation time and packet size
    std::string trafficType = "Normal";
    if (now >= 8.0 && now < 14.0 && size > 600) {
        trafficType = "Attack";
    }
    
    g_traceFile << now << "," << size << "," << trafficType << "\n";
}

int main(int argc, char *argv[])
{
    CommandLine cmd;
    cmd.Parse(argc, argv);

    Time::SetResolution(Time::NS);
    LogComponentEnable("CybersecurityPostureNs3Sim", LOG_LEVEL_INFO);

    // 1. Create 3 Nodes: Benign (0), Attacker (1), Victim (2)
    NodeContainer nodes;
    nodes.Create(3);

    NodeContainer benignLinkNodes = NodeContainer(nodes.Get(0), nodes.Get(2));
    NodeContainer attackLinkNodes = NodeContainer(nodes.Get(1), nodes.Get(2));

    // 2. Configure Point-to-Point Links: 10 Mbps, 2 ms Delay
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("2ms"));

    NetDeviceContainer benignDevices = p2p.Install(benignLinkNodes);
    NetDeviceContainer attackDevices = p2p.Install(attackLinkNodes);

    // 3. Install Internet Stack
    InternetStackHelper stack;
    stack.Install(nodes);

    // 4. Assign IP Addresses
    Ipv4AddressHelper address;
    address.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer benignInterfaces = address.Assign(benignDevices);

    address.SetBase("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer attackInterfaces = address.Assign(attackDevices);

    // 5. Install Packet Sink on Victim Node (port 5000)
    uint16_t sinkPort = 5000;
    Address sinkAddress(InetSocketAddress(Ipv4Address::GetAny(), sinkPort));
    PacketSinkHelper packetSinkHelper("ns3::UdpSocketFactory", sinkAddress);
    ApplicationContainer sinkApps = packetSinkHelper.Install(nodes.Get(2));
    sinkApps.Start(Seconds(0.0));
    sinkApps.Stop(Seconds(20.0));

    // 6. Benign Application: 100 kbps, 512B packets, continuous (0 - 20 s)
    OnOffHelper benignApp("ns3::UdpSocketFactory", InetSocketAddress(benignInterfaces.GetAddress(1), sinkPort));
    benignApp.SetConstantRate(DataRate("100kbps"), 512);
    ApplicationContainer benignApps = benignApp.Install(nodes.Get(0));
    benignApps.Start(Seconds(0.0));
    benignApps.Stop(Seconds(20.0));

    // 7. Attacker Application: 5 Mbps, 1024B packets, DoS burst (8 - 14 s)
    OnOffHelper attackApp("ns3::UdpSocketFactory", InetSocketAddress(attackInterfaces.GetAddress(1), sinkPort));
    attackApp.SetConstantRate(DataRate("5Mbps"), 1024);
    ApplicationContainer attackApps = attackApp.Install(nodes.Get(1));
    attackApps.Start(Seconds(8.0));
    attackApps.Stop(Seconds(14.0));

    // 8. Open CSV trace file and connect Rx trace sink
    g_traceFile.open("ns3_packet_trace.csv");
    g_traceFile << "timestamp,packet_size,label\n";

    Config::Connect("/NodeList/2/ApplicationList/*/$ns3::PacketSink/Rx", MakeCallback(&PacketRxTrace));

    NS_LOG_INFO("Starting 20-second ns-3 simulation...");
    Simulator::Stop(Seconds(20.0));
    Simulator::Run();
    Simulator::Destroy();

    g_traceFile.close();
    NS_LOG_INFO("Simulation complete. Trace saved to ns3_packet_trace.csv");

    return 0;
}
