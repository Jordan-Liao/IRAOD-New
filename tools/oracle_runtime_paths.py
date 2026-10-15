"""Standalone LoRA path binding for the selected221 host; stdlib only."""

import socket


TARGET_HOST = "73F3-5xA6000-221"
PREFIXES = (
    ("/mnt/shared/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"),
    ("/mnt/shared/zechuan/iraod_weights", "/home/zechuan/iraod_weights"),
    ("/mnt/HDD_14TB/zechuan/iraod_artifacts", "/home/zechuan/iraod_artifacts"),
)


def is_target_host():
    return socket.gethostname() == TARGET_HOST


def map_path(value):
    value = str(value)
    if is_target_host():
        for source, target in PREFIXES:
            if value == source or value.startswith(source + "/"):
                return target + value[len(source):]
    return value
