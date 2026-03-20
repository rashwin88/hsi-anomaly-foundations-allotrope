"""
Extracts bounding box from an EnMAP L2A scene folder
by parsing the METADATA.XML sidecar file.
"""

import os
import xml.etree.ElementTree as ET
from typing import List


def get_enmap_bounding_box(folder_path: str) -> List[float]:
    """
    Extracts bounding box from EnMAP METADATA.XML.
    Reads the tile-level <boundingPolygon> from <base><spatialCoverage>.

    Args:
        folder_path: Path to the EnMAP scene folder.

    Returns:
        [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
    """
    scene_name = os.path.basename(folder_path)
    xml_path = os.path.join(folder_path, scene_name + "-METADATA.XML")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lats, lons = [], []
    for point in root.findall(".//base/spatialCoverage/boundingPolygon/point"):
        frame = point.find("frame").text
        if frame == "center":
            continue
        lats.append(float(point.find("latitude").text))
        lons.append(float(point.find("longitude").text))

    return [min(lons), min(lats), max(lons), max(lats)]
