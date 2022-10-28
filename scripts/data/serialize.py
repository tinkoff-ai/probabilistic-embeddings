import argparse
import io
import os
from collections import OrderedDict

import mxnet as mx
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from probabilistic_embeddings.io import write_yaml
from probabilistic_embeddings.dataset import DatasetCollection
from probabilistic_embeddings.dataset.common import DatasetWrapper


def parse_arguments():
    parser = argparse.ArgumentParser("Convert dataset to mxnet format")
    parser.add_argument("src", help="Source dataset root")
    parser.add_argument("dst", help="Target dataset root")
    parser.add_argument("--dataset", help="Type of the dataset (like in training config)", required=True)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=16)
    parser.add_argument("--num-workers", help="Number of loader workers", type=int, default=4)
    return parser.parse_args()


class PackedDataset(DatasetWrapper):
    def __getitem__(self, index):
        assert self.classification
        image, label = super().__getitem__(index)
        header = mx.recordio.IRHeader(0, label, index, 0)
        buf = io.BytesIO()
        image = Image.fromarray(image)
        image.save(buf, "jpeg")
        image = buf.getvalue()
        record = mx.recordio.pack(
            header,
            image
        )
        return record


def serialize_dataset(loader, idx_path, data_path):
    writer = mx.recordio.MXIndexedRecordIO(str(idx_path), str(data_path), "w")
    i = 0
    for batch in tqdm(loader):
        for record in batch:
            writer.write_idx(i, record)
            i += 1
    writer.close()


def dump_meta(dataset, meta_path):
    meta = OrderedDict([
        ("classification", dataset.classification),
        ("num_classes", dataset.num_classes),
        ("num_samples", len(dataset))
    ])
    write_yaml(meta, str(meta_path))


def dump_labels(dataset, labels_path):
    write_yaml(np.asarray(dataset.labels).tolist(), str(labels_path))


def main(args):
    config = {
        "name": args.dataset,
        "validation_fold": None,
        "add_verification_testsets": False,
        "add_lossy_testsets": False,
        "train_repeat": 1
    }
    collection = DatasetCollection(args.src, config=config)
    datasets = collection.get_datasets(train=True, transform=False)
    datasets.update(collection.get_datasets(train=False, transform=False))
    print("Datasets:", list(datasets))
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(exist_ok=True)
    for name, dataset in datasets.items():
        if not dataset.classification:
            print("Skip verification dataset", name)
            continue
        print("Serialize", name)
        dump_meta(dataset, dst / (name + ".yaml"))
        dump_labels(dataset, dst / (name + ".labels"))
        packed_dataset = PackedDataset(dataset)
        loader = torch.utils.data.DataLoader(packed_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             collate_fn=None)
        serialize_dataset(loader, dst / (name + ".idx"), dst / (name + ".rec"))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
