import sys
import os

import hashlib
import urllib
import urllib.error
import urllib.request

from typing import (
    Any,
    Dict,
    Iterator,
    Optional,
    Tuple,
)

from tqdm import tqdm

import numpy as np

import gzip
import idx2numpy

from .dataset import Dataset

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

headers = {"User-Agent": USER_AGENT}

# dataload


#### class MNIST ####
class MNIST(Dataset):
    """MNIST `http://yann.lecun.com/exdb/mnist/` Dataset.
    mnist_train = MNIST(root=dataset_dir, train=True, download=False)
    
    """

    mirrors = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]
    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    ]

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
    ) -> None:
        self.train = train
        self.root = root

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "processed")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte.gz"
        data = idx2numpy.convert_from_file(
            gzip.open(os.path.join(self.raw_folder, image_file))
        )
        # data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte.gz"
        # targets = read_label_file(os.path.join(self.raw_folder, label_file))
        targets = idx2numpy.convert_from_file(
            gzip.open(os.path.join(self.raw_folder, label_file))
        )

        return data, targets

    def _check_exists(self) -> bool:
        """Check if the dataset already exists"""
        return all(
            check_integrity(os.path.join(self.raw_folder, fname))
            for fname, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist in raw_folder already."""
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for fname, md5 in self.resources:
            for mirror in self.mirrors:
                furl = f"{mirror}{fname}"
                try:
                    print(f"Downloading {furl}")
                    download_furl(furl, self.raw_folder, filename=fname, md5=md5)
                except urllib.error.URLError as e:
                    print(f"Failed to download (trying next):\n{e}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {fname}!")

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


#### end class MNIST ####


#### download
def download_furl(
    furl: str,
    root: str,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    max_redirect_hops: int = 3,
) -> None:
    """Download a file from a url and place it in root.
    Args:
        furl (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(furl)

    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if check_integrity(fpath, md5):
        print(f"Using downloaded and verified file: {fpath}")
        return

    # download the file
    try:
        print(f"Downloading {furl} to {fpath}")
        _urlretrieve(furl, fpath)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if furl[:5] == "https":
            furl = furl.replace("https:", "http:")
            print(
                "Failed download. Trying https -> http instead. Downloading "
                + furl
                + " to "
                + fpath
            )
            _urlretrieve(furl, fpath)
        else:
            raise e
    # check integrity of downloaded file
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")


#### download end


#### check integrity
def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    """Check if the fpath exists, if yes, check the md5 of the file."""
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


#### check integrity end


def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))


def _urlretrieve(furl: str, filename: str, chunk_size: int = 1024 * 64) -> None:
    request = urllib.request.Request(furl, headers=headers)

    with urllib.request.urlopen(request) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )
