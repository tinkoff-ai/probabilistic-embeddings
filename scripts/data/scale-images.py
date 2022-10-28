import argparse
import multiprocessing as mp
import os
import shutil

from pathlib import Path
from torchvision.transforms import Resize, CenterCrop, Compose
from tqdm import tqdm
from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPEG"}


def parse_arguments():
    parser = argparse.ArgumentParser("Scale dataset images")
    parser.add_argument("src", help="Images root")
    parser.add_argument("dst", help="Target root")
    parser.add_argument("--center-crop", help="Crop output images to center", action="store_true")
    parser.add_argument("--size", help="Size of the minimal side or coma-separated width and height", required=True)
    parser.add_argument("--num-workers", help="Number of workers", type=int, default=8)
    return parser.parse_args()


def parse_size(size):
    if "," in size:
        width, height = map(int, size.split(","))
        return (height, width)
    return int(size)


class Worker:
    def __init__(self, src, dst, image_size, crop):
        self._src = src
        self._dst = dst
        transforms = [Resize(image_size)]
        if crop:
            transforms.append(CenterCrop(image_size))
        self._transform = Compose(transforms)

    def __call__(self, src_path):
        if not os.path.isfile(src_path):
            return
        dst_path = self._dst / src_path.relative_to(self._src)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix.lower() not in EXTENSIONS:
            shutil.copy2(src_path, dst_path)
            return
        with Image.open(src_path) as image:
            image = self._transform(image)
            image.save(dst_path)


def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(exist_ok=True)

    pool = mp.Pool(args.num_workers)
    worker = Worker(src, dst, parse_size(args.size), args.center_crop)
    batch_size = args.num_workers * 8
    src_paths = list(src.rglob("*"))
    for i in tqdm(list(range(0, len(src_paths), batch_size))):
        pool.map(worker, src_paths[i:i + batch_size])


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
