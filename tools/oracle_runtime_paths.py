"""Standalone LoRA path binding for the selected221 and134 hosts; stdlib only."""

import socket


TARGET_HOST = "73F3-5xA6000-221"
TARGET_HOST_134 = "73F3-8x4090-134"
PREFIXES = (
    ("/mnt/shared/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"),
    ("/mnt/shared/zechuan/iraod_weights", "/home/zechuan/iraod_weights"),
    ("/mnt/HDD_14TB/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"),
)


def is_target_host():
    return socket.gethostname() in (TARGET_HOST, TARGET_HOST_134)


def map_path(value):
    value = str(value)
    if is_target_host():
        for source, target in PREFIXES:
            if value == source or value.startswith(source + "/"):
                return target + value[len(source):]
    return value
