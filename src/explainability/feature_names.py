"""
Feature names used by CyberAgent.

The order MUST match the order used in the final
processed network + DNS arrays.
"""


NETWORK_FEATURES = [
    "inter_arrival_time",
    "protocol_distribution",
    "packet_rate",
    "flow_duration",
    "packet_size",
]


DNS_FEATURES = [
    "query_type_distribution",
    "dns_query_frequency",
    "query_rate",
    "dns_response_activity",
    "query_length",
]


ALL_FEATURES = NETWORK_FEATURES + DNS_FEATURES


ATTACK_CLASSES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]