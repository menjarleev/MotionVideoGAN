import torch.utils.data as data

from PIL import Image
import os
import os.path


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_grouped_dataset(dir):
    images =[]
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        paths = []
        for fname in sorted(fnames):
            if is_image_file(fname):
                paths.append(os.path.join(root, fname))
        if len(paths) > 0:
            images.append(paths)
    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

def check_path_valid(A_paths, B_paths):
    assert(len(A_paths) == len(B_paths))
    for a, b in zip(A_paths, B_paths):
        assert(len(a) == len(b))

