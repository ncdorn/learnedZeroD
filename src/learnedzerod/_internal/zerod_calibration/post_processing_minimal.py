"""Minimal subset of post_processing: CSV reader for NN feature pipeline."""

import csv
import os


def read_zerod_csv(csv_path):
    """
    Read 0D simulation results from CSV.
    Handles both 'location' and 'name' as the vessel identifier column.
    """
    import numpy as np

    results = {}
    times = set()

    if not os.path.exists(csv_path):
        return results, sorted(times)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            return results, sorted(times)

        vessel_col = None
        if "location" in fieldnames:
            vessel_col = "location"
        elif "name" in fieldnames:
            vessel_col = "name"
        else:
            return results, sorted(times)

        for row in reader:
            location = row[vessel_col]
            time = float(row["time"])
            times.add(time)

            if location not in results:
                results[location] = {}
            if time not in results[location]:
                results[location][time] = {}

            for key, value in row.items():
                if key not in [vessel_col, "time"]:
                    try:
                        results[location][time][key] = float(value)
                    except (ValueError, TypeError):
                        continue

    return results, sorted(times)
