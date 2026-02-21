"""
Simple client for USGS M2M APIs
"""

import os
from typing import Dict, List
import logging
import random
import pprint
import time

from dotenv import load_dotenv
import requests
import boto3
from tqdm import tqdm

import app.utils.external_apis.usgs_m2m_filtration_templates as templates

# Load all env variables so credentials are available early.
load_dotenv()


M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable/"
S3_LOCATION = "allotrope-raw-data-india"

# Module-scoped logger; callers can configure handlers/formatters as needed.
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

        # Read credentials from the environment (via dotenv).
        self._token = os.getenv("M2M_TOKEN")
        self._username = os.getenv("M2M_USERNAME")
        self._api_key = None
        logger.info("Initializing M2M client")
        logger.debug("Username present: %s", bool(self._username))
        logger.debug("Token present: %s", bool(self._token))
        self.get_api_key()
        self._filters = None

    def get_api_key(self) -> None:
        """
        Gets the API Key
        """

        logger.info("Requesting M2M API key via login-token")
        response = requests.post(
            f"{M2M_URL}login-token",
            json={"username": self._username, "token": self._token},
            timeout=200,
        )
        logger.debug("Login response status: %s", response.status_code)
        self._api_key = response.json()["data"]
        logger.info("Received M2M API key")

    def logout(self) -> None:
        """
        Logs out of the session
        """
        logger.info("Logging out of M2M session")
        response = requests.post(
            f"{M2M_URL}logout",
            headers={"X-Auth-Token": self._api_key},
            timeout=300,
        )
        logger.debug("Logout response status: %s", response.status_code)
        print(response.json())

    def logout_and_refresh(self):
        """
        Logs out and refreshes the API key
        """
        logger.info("Refreshing M2M API key")
        self.logout()
        self.get_api_key()

    def get_dataset_filters(self, dataset_name: str = "landsat_ot_c2_l2") -> None:
        """
        Fetches all searchable fields and their hex IDs for a dataset.
        """
        logger.info("Fetching dataset filters for %s", dataset_name)
        payload = {"datasetName": dataset_name}
        headers = {"X-Auth-Token": self._api_key}
        response = requests.post(
            f"{M2M_URL}dataset-filters", json=payload, headers=headers, timeout=300
        )
        logger.debug("Dataset filters response status: %s", response.status_code)
        self._filters = response.json().get("data", [])
        logger.info("Loaded %s dataset filters", len(self._filters))

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
        max_results: int = 10,
    ):
        """
        Searches for scenes. The polygon is pre-defined. The additional filtration templates can be imported to meet specific standards
        """

        # Generate a payload for scene-search.
        # TODO: Can potentially add a cloud cover filter too.
        logger.info("Searching scenes in dataset %s", dataset_name)
        logger.debug("Search window: %s -> %s", start_date, end_date)
        logger.debug("Cloud cover min/max: %s/%s", cloud_cover_min, cloud_cover_max)
        logger.debug("Starting number: %s, max results: %s", start_num, max_results)

        # Construct the scene filtration.
        scene_filter = {}
        scene_filter["spatialFilter"] = polygon
        scene_filter["acquisitionFilter"] = {"start": start_date, "end": end_date}
        scene_filter["cloudCoverFilter"] = {
            "min": cloud_cover_min,
            "max": cloud_cover_max,
        }
        if additional_filtration_templates is not None:
            logger.debug("Applying additional filtration templates")
            scene_filter["metadataFilter"] = additional_filtration_templates

        # Build out the call payload.
        payload = {
            "datasetName": dataset_name,
            "maxResults": max_results,
            "startingNumber": start_num,
            "sceneFilter": scene_filter,
        }
        logger.info("Call Payload: %s", payload)
        headers = {"X-Auth-Token": self._api_key}
        response = requests.post(
            f"{M2M_URL}scene-search", json=payload, headers=headers, timeout=300
        )
        logger.debug("Scene search response status: %s", response.status_code)
        return response.json().get("data")

    def build_download_options(self, entity_id: str, dataset: str = "landsat_ot_c2_l2"):
        """
        Builds a set of download options for a given entity id. The download options call produces a list of files for download.

        Args:
        entity_id (str) : The entity ID that is going to be downloaded
        """
        logger.info("Building download options for entity %s", entity_id)
        headers = {"X-Auth-Token": self._api_key}
        payload = {"datasetName": dataset, "entityIds": [entity_id]}

        response = requests.post(
            f"{M2M_URL}download-options", json=payload, headers=headers, timeout=300
        )
        logger.debug("Download options response status: %s", response.status_code)
        return response.json().get("data")

    def download_request(self, downloads: List[Dict[str, str]], label: str) -> Dict:
        """
        Creates a download request
        """
        logger.info("Creating download request for %s items", len(downloads))
        headers = {"X-Auth-Token": self._api_key}
        payload = {"downloads": downloads, "label": label}

        # Fire the request
        response = requests.post(
            f"{M2M_URL}download-request", json=payload, headers=headers, timeout=300
        )
        logger.debug("Download request response status: %s", response.status_code)
        return response.json().get("data")


class M2MSampler:
    """
    Given a payload, can sample in a specific proportion for scenes.

    The idea is simple. Run a probe query to identify the number of hits,
    then identify the different starting numbers and pick out only those datasets.
    """

    def __init__(
        self,
        start_date: str,
        end_date: str,
        dataset_name: str = "landsat_ot_c2_l2",
        additional_filtration_templates: Dict = None,
        cloud_cover_min: int = 0,
        cloud_cover_max: int = 100,
    ):
        """
        Initializes the class.
        """

        # Set up the client and query parameters.
        self.client = M2MClient()
        self.start_date = start_date
        self.end_date = end_date
        self.dataset_name = dataset_name
        self.additional_filtration_templates = additional_filtration_templates
        self.cc_min = cloud_cover_min
        self.cc_max = cloud_cover_max
        self.polygon = self.get_india_bb()

        # Load up the S3 client used for uploading downloads.
        self.s3_client = boto3.client("s3", region_name="ap-south-1")
        logger.info(
            "Initialized sampler for %s, %s -> %s",
            self.dataset_name,
            self.start_date,
            self.end_date,
        )

    def get_india_bb(self) -> Dict:
        """
        Gets the bounding box for India. We will be sampling from India here
        # TODO: We can make this a bit more modular in the future.
        """

        logger.debug("Using hard-coded India bounding box polygon")
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

    def run_probe(self) -> int:
        """
        Runs a probing query to figure out how many results match the given criteria
        """
        logger.info("Running probe query to count results")
        try:
            probe = self.client.search_scenes(
                polygon=self.polygon,
                start_date=self.start_date,
                end_date=self.end_date,
                start_num=1,
                dataset_name=self.dataset_name,
                additional_filtration_templates=self.additional_filtration_templates,
                cloud_cover_min=self.cc_min,
                cloud_cover_max=self.cc_max,
            )
            results = probe.get("totalHits")
            logger.info("Probe query returned %s total hits", results)
            return results
        except Exception as err:
            logger.exception("Probe query failed")
            raise Exception(
                f"Some error occured when running the probing operation for the given search criteria: {str(err)}"
            ) from err

    def generate_samples(
        self, result_count: int, sampling_percentage: float = 0.4
    ) -> List[int]:
        """
        Simple way to generate samples from the number of results
        """

        logger.info(
            "Generating samples from %s results at %s%%",
            result_count,
            sampling_percentage * 100,
        )
        sample_count = int(result_count * sampling_percentage)
        logger.debug("Computed sample count: %s", sample_count)
        samples = random.sample(range(1, result_count + 1), k=sample_count)
        logger.info("Generated %s sample indices", len(samples))
        return samples

    def download_single_sample(self, sample_number: int) -> None:
        """
        Downloads the chosen files from a given sample.

        There are a very specific set of files I want to download in this case. So will just choose them.
        """

        logger.info("Downloading sample %s", sample_number)
        # First we need to get the entity ID that we want to download.
        entity_search = self.client.search_scenes(
            polygon=self.polygon,
            start_date=self.start_date,
            end_date=self.end_date,
            start_num=sample_number,
            dataset_name=self.dataset_name,
            additional_filtration_templates=self.additional_filtration_templates,
            cloud_cover_min=self.cc_min,
            cloud_cover_max=self.cc_max,
            max_results=1,
        )

        entity_id = entity_search.get("results")[0].get("entityId")
        logger.info("Found entity id: %s which is sample: %s", entity_id, sample_number)

        # Generate download-options.
        download_options = self.client.build_download_options(
            entity_id=entity_id, dataset=self.dataset_name
        )
        logger.info("Generated download options")

        # Build out a list of all files and select from that.
        all_files = []
        for bundle in download_options:
            secondary = bundle.get("secondaryDownloads", [])
            if secondary:
                all_files.extend(secondary)
        logger.debug("Found %s secondary files", len(all_files))

        unique_files = {}
        # Make everything unique by entity id.
        for file in all_files:
            unique_files[file.get("entityId")] = {
                "downloadId": file.get("id"),
                "entityId": file.get("entityId"),
                "displayId": file.get("displayId"),
            }
        logger.debug("Unique file entries: %s", len(unique_files))

        # File patterns to match for download.
        pattern_match = lambda file: file.endswith("ST_B10.TIF") or file.endswith(
            "QA_PIXEL.TIF"
        )

        # Get the downloadables matching desired patterns.
        downloadables = [
            downloadable
            for key, downloadable in unique_files.items()
            if pattern_match(downloadable.get("displayId"))
        ]
        logger.info("Collected %s downloadables", len(downloadables))

        # Form the download request payload.
        download_response = self.client.download_request(
            downloads=[
                {
                    "downloadId": file.get("downloadId"),
                    "entityId": file.get("entityId"),
                    "productId": file.get("downloadId"),
                }
                for file in downloadables
            ],
            label=entity_id,
        )

        # Collect the available downloads.
        available_downloads = download_response.get("availableDownloads")
        logger.debug("Available downloads: %s", len(available_downloads or []))

        # Attach download URLs to the requested items.
        for download in available_downloads:
            entity = download.get("entityId")
            for downloadable in downloadables:
                if downloadable.get("entityId") == entity:
                    downloadable["url"] = download.get("url")
                    break

        # Now we can form a base S3 key for all uploads.
        s3_base_key = f"landsat/{entity_id}/"

        for downloadable in downloadables:
            dl_url = downloadable.get("url")
            logger.info("Downloading %s", downloadable.get("displayId"))
            with requests.get(dl_url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                s3_key = f"{s3_base_key}{downloadable.get('displayId')}"
                logger.debug("Uploading to s3://%s/%s", S3_LOCATION, s3_key)
                # tqdm bar for this specific file
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=s3_key
                ) as pbar:
                    # We use a Callback to update tqdm as boto3 reads from the stream
                    self.s3_client.upload_fileobj(
                        r.raw,
                        S3_LOCATION,
                        s3_key,
                        Callback=lambda bytes_transferred: pbar.update(
                            bytes_transferred
                        ),
                    )

    def orchestrate_pull(self):
        """
        Orchestrates a full datapull
        """
        # First we run a probe
        hits = self.run_probe()
        logger.info("Hits from probe run: %s", hits)

        # Generate samples
        samples = self.generate_samples(result_count=hits, sampling_percentage=0.3)
        logger.info("Generated %s Samples", len(samples))

        # Start downloading
        for item in tqdm(samples, desc="Scenes Processed"):
            self.download_single_sample(item)
            time.sleep(3)
            self.client.logout_and_refresh()


if __name__ == "__main__":

    # Configure root logger to show all logs (DEBUG and above) in the console.
    import sys

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console_handler.setFormatter(formatter)
    # Remove any old handlers first (avoid duplicate logs in some environments)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    try:
        logger.info("Running sample download flow via __main__")
        sampler = M2MSampler(
            start_date="2025-01-01",
            end_date="2025-02-01",
            cloud_cover_max=80,
            cloud_cover_min=10,
            additional_filtration_templates=templates.LC09_L2SP,
        )
        sampler.orchestrate_pull()
    finally:
        logger.info("Logging out after sample run")
        sampler.client.logout()
