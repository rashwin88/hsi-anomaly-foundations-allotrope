"""
Simple client for USGS M2M APIs
"""

import json
import os
from typing import Dict
import logging

from dotenv import load_dotenv
import requests

import app.utils.external_apis.usgs_m2m_filtration_templates as templates

# Load all env variables
load_dotenv()


M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"


logger = logging.getLogger("M2MAPI")
logger.setLevel(logging.INFO)


class M2MClient:
    """
    Simple Client for different M2M API functionalities.
    Provides some lower level functionalities which can be packaged into higher level functionalities.
    """

    def __init__(self):
        """
        Constructor
        """

        # Set up all the required keys
        self._token = os.getenv("M2M_TOKEN")
        self._username = os.getenv("M2M_USERNAME")
        self._api_key = None
        self.get_api_key()
        self._filters = None

    def get_api_key(self) -> None:
        """
        Gets the API Key
        """

        response = requests.post(
            f"{M2M_URL}login-token",
            json={"username": self._username, "token": self._token},
            timeout=200,
        )
        self._api_key = response.json()["data"]

    def logout(self) -> None:
        """
        Logs out of the session
        """
        response = requests.post(
            f"{M2M_URL}logout",
            headers={"X-Auth-Token": self._api_key},
            timeout=300,
        )
        print(response.json())

    def logout_and_refresh(self):
        """
        Logs out and refreshes the API key
        """
        self.logout()
        self.get_api_key()

    def get_dataset_filters(self, dataset_name: str = "landsat_ot_c2_l2") -> None:
        """
        Fetches all searchable fields and their hex IDs for a dataset.
        """
        payload = {"datasetName": dataset_name}
        headers = {"X-Auth-Token": self._api_key}
        response = requests.post(
            f"{M2M_URL}dataset-filters", json=payload, headers=headers, timeout=300
        )
        self._filters = response.json().get("data", [])

    def search_scenes(
        self,
        polygon: Dict,
        start_date: str,
        end_date: str,
        start_num=1,
        dataset_name: str = "landsat_ot_c2_l2",
        additional_filtration_templates: Dict = None,
        cloud_cover_min: int = 0,
        cloud_cover_max: int = 100,
    ):
        """
        Searches for scenes. The polygon is pre-defined. The additional filtration templates can be imported to meet specific standards
        """

        # Generate a payload
        ## TODO: Can potentially add a cloud cover filter too.

        # Construct the scene filtration
        scene_filter = {}
        scene_filter["spatialFilter"] = polygon
        scene_filter["acquisitionFilter"] = {"start": start_date, "end": end_date}
        scene_filter["cloudCoverFilter"] = {
            "min": cloud_cover_min,
            "max": cloud_cover_max,
        }
        if additional_filtration_templates is not None:
            scene_filter["metadataFilter"] = additional_filtration_templates

        # Build out the call payload
        payload = {
            "datasetName": dataset_name,
            "maxResults": 10,
            "startingNumber": start_num,
            "sceneFilter": scene_filter,
        }
        logger.info("Call Payload: %s", payload)
        headers = {"X-Auth-Token": self._api_key}
        response = requests.post(
            f"{M2M_URL}scene-search", json=payload, headers=headers, timeout=300
        )
        return response.json()

    def build_download_options(self, entity_id: str, dataset: str = "landsat_ot_c2_l2"):
        """
        Builds a set of download options for a given entity id
        """
        self.get_api_key()
        headers = {"X-Auth-Token": self._api_key}
        payload = {"datasetName": dataset, "entityIds": [entity_id]}

        response = requests.post(
            f"{M2M_URL}download-options", json=payload, headers=headers, timeout=300
        )
        return response.json()

    def get_india_bb(self) -> Dict:
        """
        Gets the bounding box for India
        """

        return {
            "filterType": "geojson",
            "geoJson": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [68.17, 7.96],
                        [97.40, 7.96],
                        [97.40, 35.49],
                        [68.17, 35.49],
                        [68.17, 7.96],
                    ]
                ],
            },
        }


class M2MSampler:
    """
    Given a payload, can sample in a specific proportion for scenes.

    The idea is simple. Run a probe query to identify the number of hits,
    then identify the different starting numbers and pick out only those datasets.
    """

    def __init__(self):
        """
        Initializes the class.
        """

        # Set up the client
        self.client = M2MClient()


if __name__ == "__main__":

    try:
        client = M2MClient()
        client.logout_and_refresh()
        results = client.build_download_options(entity_id="LC91450352024005LGN00")
        with open("options.json", "w") as f:
            json.dump(results, f, indent=2)
    finally:
        client.logout()
