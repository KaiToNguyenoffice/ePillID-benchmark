"""
Incremental LabelEncoder utilities.

Provides ``IncrementalLabelEncoder`` which extends sklearn's LabelEncoder
to support adding new classes without re-fitting from scratch, and a
helper ``load_or_create_encoder`` used by the training scripts.
"""

import os
import pickle
import logging

import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class IncrementalLabelEncoder:
    """Wraps sklearn LabelEncoder with incremental class support.

    * ``partial_fit(labels)`` adds new classes while preserving existing indices.
    * ``transform`` / ``inverse_transform`` work the same as sklearn.
    * Compatible with pickle serialization.
    """

    def __init__(self, base_encoder=None):
        if base_encoder is not None:
            self._classes = list(base_encoder.classes_)
        else:
            self._classes = []
        self._label_to_idx = {c: i for i, c in enumerate(self._classes)}

    @property
    def classes_(self):
        return np.array(self._classes)

    def fit(self, labels):
        unique = sorted(set(labels))
        self._classes = list(unique)
        self._label_to_idx = {c: i for i, c in enumerate(self._classes)}
        return self

    def partial_fit(self, labels):
        """Add new classes while keeping existing index assignments."""
        new_classes = sorted(set(labels) - set(self._classes))
        if new_classes:
            logger.info("Adding %d new classes: %s...", len(new_classes),
                        new_classes[:5])
            for c in new_classes:
                self._label_to_idx[c] = len(self._classes)
                self._classes.append(c)
        return self

    def transform(self, labels):
        return np.array([self._label_to_idx[l] for l in labels], dtype=np.int64)

    def inverse_transform(self, indices):
        return np.array([self._classes[i] for i in indices])

    def __len__(self):
        return len(self._classes)


def load_or_create_encoder(encoder_path, all_imgs_csv=None, label_col='label',
                           incremental=False):
    """Load encoder from disk, or create + save a new one.

    When ``incremental=True`` and an existing encoder is found, any new classes
    in the CSV are appended (indices preserved for existing classes).

    Returns an ``IncrementalLabelEncoder``.
    """
    if not os.path.exists(encoder_path):
        alt = _find_alternate_encoder(encoder_path)
        if alt:
            logger.info("Encoder not at %s, found at %s", encoder_path, alt)
            encoder_path = alt

    loaded = False
    if os.path.exists(encoder_path):
        logger.info("Loading label encoder: %s", encoder_path)
        try:
            with open(encoder_path, 'rb') as f:
                raw = pickle.load(f)

            if isinstance(raw, IncrementalLabelEncoder):
                encoder = raw
            else:
                encoder = IncrementalLabelEncoder(base_encoder=raw)
            loaded = True

            if incremental and all_imgs_csv is not None:
                import pandas as pd
                df = pd.read_csv(all_imgs_csv)
                old_count = len(encoder)
                encoder.partial_fit(df[label_col].values)
                if len(encoder) > old_count:
                    logger.info("Encoder expanded from %d to %d classes",
                                old_count, len(encoder))
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(encoder, f)
        except (ModuleNotFoundError, ImportError, pickle.UnpicklingError) as exc:
            logger.warning(
                "Could not load encoder from %s (%s). "
                "Will re-create from CSV.", encoder_path, exc)
            loaded = False

    if not loaded:
        if all_imgs_csv is None:
            raise FileNotFoundError(
                f"No encoder at {encoder_path} and no CSV provided to create one")

        logger.warning("Fitting a new label encoder at %s", encoder_path)
        import pandas as pd
        df = pd.read_csv(all_imgs_csv)
        encoder = IncrementalLabelEncoder()
        encoder.fit(df[label_col].values)

        os.makedirs(os.path.dirname(encoder_path) or '.', exist_ok=True)
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)

    logger.info("Label encoder has %d classes", len(encoder))
    return encoder


def _find_alternate_encoder(encoder_path):
    """Try common alternate filenames for the label encoder pickle."""
    directory = os.path.dirname(encoder_path)
    if not os.path.isdir(directory):
        return None

    alternates = ['label_encoder.pickle', 'label_encoder_pytorch131.pickle']
    for alt_name in alternates:
        alt_path = os.path.join(directory, alt_name)
        if os.path.exists(alt_path) and alt_path != encoder_path:
            return alt_path
    return None
