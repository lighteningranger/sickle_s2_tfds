import multiprocessing
import tensorflow as tf
import numpy as np
import zarr
from pathlib import Path
import pickle
from tqdm import tqdm
from functools import partial

DATA_DIR = "/home/moti/data/sickle_s2.zarr"


def get_month_year(dates):
    """Convert dates to month and year"""
    return dates.astype("datetime64[M]").astype(int) % 12, dates.astype(
        "datetime64[Y]"
    ).astype(int) + 1970


def create_date_idxs(dates):
    """Create valid date index pairs with difference <= 3 months"""
    n = len(dates)
    indices = np.triu_indices(n)
    m1, y1 = get_month_year(dates[indices[0]])
    m2, y2 = get_month_year(dates[indices[1]])
    date_diff = m2 - m1 + 12 * (y2 - y1)
    valid_mask = (date_diff >= 0) & (date_diff <= 3)
    valid_indices = np.column_stack(indices)[valid_mask]
    return valid_indices


def build_indexes(ds, num_samples=6602, cache_file="indexes.npy"):
    """Build and cache dataset indexes"""
    if Path(cache_file).exists():
        return np.load(cache_file)

    indexes = []
    ids = range(num_samples)

    for _id in tqdm(ids):
        try:
            ts = ds[f"sample_{_id}"]["timestamps"][:]
            valid_idxs = create_date_idxs(ts)
            indexes.extend([[_id, t1, t2] for t1, t2 in valid_idxs])
        except KeyError:
            continue
    indexes = np.array(indexes, dtype=np.int64)
    np.save(cache_file, indexes)
    return indexes


@tf.numpy_function(
    Tout=[tf.float64, tf.float64, tf.int64, tf.int64, tf.int64, tf.int64]
)
def load_data(idx):
    """Load data for given sample ID and timestamps"""
    _id, t1, t2 = idx
    ds = zarr.open(DATA_DIR)
    sample = ds[f"sample_{_id}"]
    chip1 = sample["data"][t1, ...][:]
    chip2 = sample["data"][t2, ...][:]
    chip1 = tf.convert_to_tensor(chip1, dtype=tf.float64)
    chip2 = tf.convert_to_tensor(chip2, dtype=tf.float64)
    t1 = sample["timestamps"][t1]
    t2 = sample["timestamps"][t2]
    month1, year1 = get_month_year(t1)
    month1, year1 = tf.convert_to_tensor(month1), tf.convert_to_tensor(year1)
    month2, year2 = get_month_year(t2)
    month2, year2 = tf.convert_to_tensor(month2), tf.convert_to_tensor(year2)
    return chip1, chip2, month1, month2, year1, year2


def create_dataset(data_dir, num_samples=6602, cache_file="indexes.npy", seed=42):
    ds = zarr.open(data_dir)
    indexes = build_indexes(ds=ds, num_samples=num_samples, cache_file=cache_file)
    ds.store.close()

    # Train - Val Split
    np.random.RandomState(seed).shuffle(indexes)
    split_idx = int(0.8 * len(indexes))
    train_indexes = indexes[:split_idx]
    val_indexes = indexes[split_idx:]
    tf_train_ds = tf.data.Dataset.from_tensor_slices(train_indexes).map(
        load_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    tf_val_ds = tf.data.Dataset.from_tensor_slices(val_indexes).map(
        load_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )

    train_ds = tf_train_ds#.prefetch(tf.data.AUTOTUNE)
    val_ds = tf_val_ds#.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = create_dataset(DATA_DIR)
    for batch_idx, batch in enumerate(tqdm(train_ds)):
        chip1, chip2, month1, year1, month2, year2 = batch
        # if batch_idx % 5 == 0:
        #     print("Chip1 shape:", chip1.shape)
        #     print("Chip2 shape:", chip2.shape)
        #     print("Month1 shape:", month1.shape)
        #     print("Year1 shape:", year1.shape)
        #     print("Month2 shape:", month2.shape)
        #     print("Year2 shape:", year2.shape)
        breakpoint()
