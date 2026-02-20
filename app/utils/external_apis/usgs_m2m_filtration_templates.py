"""
Templates for metadata filtering in the M2M API
"""

LC09_L2SP = {
    "filterType": "and",
    "childFilters": [
        {
            # Filter for Landsat 9 specifically
            "filterType": "value",
            "filterId": "61af9273566bb9a8",
            "value": "9",
            "operand": "=",
        },
    ],
}
