"""Dataset class order shared by native result and adapter metadata consumers."""

CLASSES = {
    "RSAR": ("ship", "aircraft", "car", "tank", "bridge", "harbor"),
    "DIOR": ("airplane", "airport", "baseballfield", "basketballcourt", "bridge",
             "chimney", "expressway-service-area", "expressway-toll-station", "dam",
             "golffield", "groundtrackfield", "harbor", "overpass", "ship", "stadium",
             "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"),
}

ORACLE_TRAINING_TEMPLATES = {
    "RSAR": "A SAR image of a {}",
    "DIOR": "an aerial image of a {}",
}
