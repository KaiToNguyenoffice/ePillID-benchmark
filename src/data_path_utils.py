"""
Data path validation and auto-correction utilities.

Handles common issues such as double-nested directories that occur when
zip files contain a root folder (e.g. ePillID_data.zip → ePillID_data/ePillID_data/).
"""

import os
import logging

logger = logging.getLogger(__name__)

REQUIRED_SUBDIRS = ['folds', 'classification_data']


def validate_data_root(data_root_dir, auto_fix=True):
    """Validate that data_root_dir contains the expected subdirectories.

    If the expected layout is found one level deeper (double-nested zip
    extraction), the corrected path is returned when ``auto_fix=True``.

    Returns the validated (possibly corrected) data_root_dir.

    Raises FileNotFoundError if the data cannot be located.
    """
    data_root_dir = os.path.normpath(data_root_dir)

    if _has_required_subdirs(data_root_dir):
        logger.info("Data root validated: %s", data_root_dir)
        return data_root_dir

    if auto_fix:
        for entry in os.listdir(data_root_dir) if os.path.isdir(data_root_dir) else []:
            candidate = os.path.join(data_root_dir, entry)
            if os.path.isdir(candidate) and _has_required_subdirs(candidate):
                logger.warning(
                    "Double-nested data directory detected. "
                    "Auto-correcting data_root_dir:\n"
                    "  provided : %s\n"
                    "  corrected: %s",
                    data_root_dir, candidate)
                return candidate

    missing = [d for d in REQUIRED_SUBDIRS
               if not os.path.isdir(os.path.join(data_root_dir, d))]

    raise FileNotFoundError(
        f"data_root_dir '{data_root_dir}' is missing required subdirectories: "
        f"{missing}.\n"
        f"Expected structure:\n"
        f"  {data_root_dir}/\n"
        f"  ├── classification_data/   (pill images)\n"
        f"  ├── folds/                 (CSV split files)\n"
        f"  └── ...\n\n"
        f"Hint: If you extracted a zip file, check whether the data is one "
        f"level deeper (e.g. ePillID_data/ePillID_data/)."
    )


def _has_required_subdirs(path):
    if not os.path.isdir(path):
        return False
    return all(
        os.path.isdir(os.path.join(path, subdir))
        for subdir in REQUIRED_SUBDIRS
    )
