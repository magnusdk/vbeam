import os
import time
from pathlib import PurePosixPath
from typing import Sequence, Union
from urllib.parse import urlparse, urlsplit
from urllib.request import urlretrieve


class DownloadReporter:
    """Class for printing the progress of downloading a file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.throttle_s = 0.1
        self.prev_progress = time.time() - self.throttle_s
        self.show_progress(0)

    def show_progress(self, percentage, final=False):
        if (time.time() - self.prev_progress) >= self.throttle_s or final:
            print(
                f'Downloading file to "{self.filepath}": {percentage:.1f}%',
                end="\r" if not final else None,
            )
            self.prev_progress = time.time()

    def __call__(self, block_num: int, read_size: int, total_file_size: int):
        percentage = min(block_num * read_size, total_file_size) / total_file_size * 100
        self.show_progress(percentage, final=block_num * read_size >= total_file_size)


def cached_download(
    url: str,
    local_dir: str = None,
    filename: Union[str, Sequence[str]] = None,
    invalidate_cache: bool = False,
) -> str:
    """
    Download and cache the file at URL, if not already cached. Return the path to the 
    file.

    Args:
      url: The URL of the file to be downloaded.
      local_dir: Optional base-directory of where to store the file. If not provided,
        either the environment variable VBEAM_DOWNLOADS or "~/.vbeam_downloads" is used.
      filename: Optional name of the downloaded file. Can be a string, or a sequence of
        strings representing the full path, e.g.:
          ("foo", "bar", "baz.txt") -> "foo/bar/baz.txt".
        If not provided, the directories and filename are inferred from the URL, e.g.:
          "https://www.example.com/foo/bar/baz.txt" -> "www.example.com/foo/bar/baz.txt"
      invalidate_cache: If True, forces (re)downloading the file from the URL.

    Returns:
      The path to the cached file downloaded from the URL.
    """
    # Read from VBEAM_DOWNLOADS environment variable if local_dir is None, or set it to
    # the default value of "~/.vbeam_downloads".
    if not local_dir:
        local_dir = os.environ.get("VBEAM_DOWNLOADS", None) or "~/.vbeam_downloads"

    local_dir = os.path.expanduser(local_dir)  # Expand tilde (~) in path
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # Convert filename to a sequence of directory names, followed by the name of the
    # file, if filename is None or a string.
    if filename is None:
        parsed_url = urlparse(url)
        filename = (parsed_url.hostname,) + PurePosixPath(urlsplit(url).path).parts[1:]
    elif isinstance(filename, str):
        filename = (filename,)

    filepath = f"{local_dir}/{'/'.join(filename)}"
    filedir = f"{local_dir}/{'/'.join(filename[:-1])}"
    if not os.path.exists(filepath) or invalidate_cache:
        os.makedirs(filedir, exist_ok=True)
        urlretrieve(url, filepath, reporthook=DownloadReporter(filepath))

    return filepath
