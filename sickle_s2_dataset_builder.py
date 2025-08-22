"""sickle_s2 dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import zarr
from dataclasses import dataclass
from typing import Tuple
from sickle_s2.data import create_dataset, DATA_DIR


@dataclass
class DefaultConfig(tfds.core.BuilderConfig):
    diffs: Tuple[int] = (0, 1, 2, 3)


_DESCRIPTION = "Sickle S2 SSL: bitemporal image pairs for SSL"


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Sickle S2 SSL dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    BUILDER_CONFIGS = [
        DefaultConfig(
            name="default", description="pairs 0,1,2,3 months apart", diffs=(0, 1, 2, 3)
        )
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "chip1": tfds.features.Tensor(
                        shape=(4, 224, 224), dtype=tf.float64
                    ),
                    "chip2": tfds.features.Tensor(
                        shape=(4, 224, 224), dtype=tf.float64
                    ),
                    "month1": tfds.features.Scalar(dtype=tf.int64),
                    "month2": tfds.features.Scalar(dtype=tf.int64),
                    "year1": tfds.features.Scalar(dtype=tf.int64),
                    "year2": tfds.features.Scalar(dtype=tf.int64),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        train_ds, val_ds = create_dataset(DATA_DIR)
        return {
            "train": self._generate_examples(train_ds),
            "test": self._generate_examples(val_ds),
        }

    def _generate_examples(self, ds):
        for i,sample in enumerate(tfds.as_numpy(ds)):
            chip1, chip2, month1, month2, year1, year2 = sample
            yield (
                i,
                {
                    "chip1": chip1,
                    "chip2": chip2,
                    "month1": month1,
                    "month2": month2,
                    "year1": year1,
                    "year2": year2,
                },
            )
