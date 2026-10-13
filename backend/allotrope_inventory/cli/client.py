"""Thin HTTP client for the inventory api.

`InventoryError` carries the api's `detail` (a code string, or a dict for
upload_incomplete / checksum_mismatch) so commands can react to it.
"""

import os
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:8020"


class InventoryError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail

    @property
    def code(self) -> str | None:
        """The error code whether `detail` is a string or a {"code": …} dict."""
        if isinstance(self.detail, dict):
            return self.detail.get("code")
        return self.detail if isinstance(self.detail, str) else None


class Client:
    def __init__(self, url: str | None = None, key: str | None = None):
        key = key or os.environ.get("INVENTORY_KEY")
        if not key:
            raise SystemExit("INVENTORY_KEY is not set")
        self._http = httpx.Client(
            base_url=(url or os.environ.get("INVENTORY_URL", DEFAULT_URL)).rstrip("/"),
            headers={"X-Inventory-Key": key},
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    def request(self, method: str, path: str, **kw: Any) -> Any:
        r = self._http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            raise InventoryError(r.status_code, detail)
        return r.json()

    def get(self, path: str, **params: Any) -> Any:
        # Drop unset filters so the api sees only what the user asked for.
        return self.request("GET", path, params={k: v for k, v in params.items() if v not in (None, [])})

    def post(self, path: str, body: Any = None, timeout: float | None = 60.0) -> Any:
        """`timeout=None` for calls that legitimately run long (server-side imports)."""
        return self.request("POST", path, json=body, timeout=timeout)

    def verified_files(self, dataset_id: str) -> dict[str, str]:
        """{relative_path: sha256} of every verified file, paging through
        /files (a collection can hold ~500k)."""
        out: dict[str, str] = {}
        offset = 0
        while True:
            page = self.get(f"/datasets/{dataset_id}/files", uploaded="true", limit=1000, offset=offset)
            out.update({f["relative_path"]: f["sha256"] for f in page["items"]})
            offset += len(page["items"])
            if not page["items"] or offset >= page["total"]:
                return out

    def put(self, path: str) -> Any:
        return self.request("PUT", path)

    def delete(self, path: str) -> Any:
        return self.request("DELETE", path)
