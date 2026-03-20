"""
Parses EnMAP L2A METADATA.XML sidecar files into structured EnmapMetadata models.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict

from app.models.file_processing.enmap_metadata import (
    EnmapMetadata,
    EnmapBandCharacterisation,
    EnmapQualityFlags,
    EnmapDetectorBoundary,
    EnmapSpatialInfo,
)


class EnmapXmlParser:
    """
    Parses an EnMAP L2A METADATA.XML file and returns a structured EnmapMetadata model.
    """

    def __init__(self, xml_path: str):
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

    def parse(self) -> EnmapMetadata:
        """Parses the full XML and returns structured metadata."""
        specific = self.root.find("specific")
        base = self.root.find("base")

        # Temporal coverage
        temporal = base.find("temporalCoverage")
        start_time = datetime.fromisoformat(temporal.find("startTime").text.rstrip("Z"))
        stop_time = datetime.fromisoformat(temporal.find("stopTime").text.rstrip("Z"))

        return EnmapMetadata(
            processing_level=base.find("level").text,
            mission=specific.find("mission").text,
            satellite_id=specific.find("satelliteID").text,
            tile_id=specific.find("tileID").text,
            datatake_id=specific.find("datatakeID").text,
            start_time=start_time,
            stop_time=stop_time,
            band_characterisation=self._parse_band_characterisation(specific),
            quality_flags=self._parse_quality_flags(specific),
            detector_boundary=self._parse_detector_boundary(specific),
            spatial_info=self._parse_spatial_info(base, specific),
        )

    def _parse_band_characterisation(
        self, specific: ET.Element
    ) -> List[EnmapBandCharacterisation]:
        """Parses the <bandCharacterisation> section for all 224 bands."""
        bands = []
        band_char = specific.find("bandCharacterisation")
        for band_el in band_char.findall("bandID"):
            bands.append(
                EnmapBandCharacterisation(
                    band_number=int(band_el.get("number")),
                    wavelength_center=float(
                        band_el.find("wavelengthCenterOfBand").text
                    ),
                    fwhm=float(band_el.find("FWHMOfBand").text),
                    gain=float(band_el.find("GainOfBand").text),
                    offset=float(band_el.find("OffsetOfBand").text),
                )
            )
        return bands

    def _parse_quality_flags(self, specific: ET.Element) -> EnmapQualityFlags:
        """Parses the <qualityFlag> section for scene-level quality metrics."""
        qf = specific.find("qualityFlag")
        return EnmapQualityFlags(
            cloud_cover=float(qf.find("cloudCover").text),
            haze_cover=float(qf.find("hazeCover").text),
            cirrus_cover=float(qf.find("cirrusCover").text),
            snow_cover=float(qf.find("snowCover").text),
            water_cover=float(qf.find("waterCover").text),
            cloud_shadow=float(qf.find("cloudShadow").text),
            dead_pixels_vnir=int(qf.find("deadPixelsVNIR").text),
            dead_pixels_swir=int(qf.find("deadPixelsSWIR").text),
        )

    def _parse_detector_boundary(self, specific: ET.Element) -> EnmapDetectorBoundary:
        """
        Parses VNIR/SWIR channel lists from vnirProductQuality and swirProductQuality,
        plus band counts from numberOfVNIRBands/numberOfSWIRBands.
        """
        vnir_pq = specific.find("vnirProductQuality")
        swir_pq = specific.find("swirProductQuality")

        vnir_channels_str = vnir_pq.find("expectedChannelsList").text
        swir_channels_str = swir_pq.find("expectedChannelsList").text

        vnir_channels = [int(x) for x in vnir_channels_str.split(",")]
        swir_channels = [int(x) for x in swir_channels_str.split(",")]

        return EnmapDetectorBoundary(
            num_vnir_bands=int(specific.find("numberOfVNIRBands").text),
            num_swir_bands=int(specific.find("numberOfSWIRBands").text),
            vnir_expected_channels=vnir_channels,
            swir_expected_channels=swir_channels,
        )

    def _parse_spatial_info(
        self, base: ET.Element, specific: ET.Element
    ) -> EnmapSpatialInfo:
        """Parses bounding polygon from <base> and dimensions from <specific>."""
        polygon_points = []
        for point in base.findall("spatialCoverage/boundingPolygon/point"):
            frame = point.find("frame").text
            if frame == "center":
                continue
            polygon_points.append(
                {
                    "frame": frame,
                    "latitude": float(point.find("latitude").text),
                    "longitude": float(point.find("longitude").text),
                }
            )

        width_el = specific.find("widthOfScene")
        height_el = specific.find("heightOfScene")

        return EnmapSpatialInfo(
            bounding_polygon=polygon_points,
            width=int(width_el.text),
            height=int(height_el.text),
            background_value=int(specific.find("backgroundValue").text),
        )
