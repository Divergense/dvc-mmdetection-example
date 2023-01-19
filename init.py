"""
Init steps:

1. Create necessary dirictories (checkpoints)

2. Install MMDetection

3. Setup configs directory of MMDetection

4. Setup device type
"""
import os

from src.utils import (
    PackageWrapperMmcv, 
    PackageWrapperPip, 
    PackageWrapperGithub
    )

from pathlib import Path
from typing import List, Tuple, Dict, Any
from yaml import safe_load, safe_dump
from dotenv import load_dotenv


load_dotenv()
PARAMS_FILE = os.environ.get("PARAMS_DATA")
DIRS = ["checkpoints"]
PACKAGES = [
    PackageWrapperMmcv('mmcv-full'), 
    PackageWrapperGithub("mmdetection", "https://github.com/Divergense/mmdetection.git"),
    ]
CWD = Path.cwd()


def load_params(params_file: str) -> Dict[str, Any]:
    if PARAMS_FILE is None:
        raise ValueError('Check .env file')

    with open(params_file, "r") as file:
        params = safe_load(file)
    return params


def create_dirs(dirs: List[str]) -> None:
    for dir in dirs:
        _dir = CWD / Path(dir)
        try:
            _dir.mkdir()
        except FileExistsError as exc:
            print(exc)


def install_packages(packages: List[Tuple]) -> None:
    """
    Param packages consists of tuples the following format:
    (package_name, github_url)
    If github_url is None package will be installed through pip,
    else github will be used.
    """
    import importlib
    
    for package in packages:
        try:
            importlib.import_module(package.name)
            print(f'Package {package.name} is already installed.')
        except ImportError:
            package.install()


def setup_mmdet(params: Dict[str, Any]) -> None:
    default_mmdet_configs = Path("mmdetection/configs/faster_rcnn/")
    if not default_mmdet_configs.is_dir():
        default_mmdet_configs = None

    mmdet_configs = input(
        f"Enter mmdetection configurations location ({default_mmdet_configs}): "
    )
    if mmdet_configs == "":
        if default_mmdet_configs is None:
            raise RuntimeError("Path to MMDetection is not set.")
        mmdet_configs = default_mmdet_configs

    mmdet_configs = Path(mmdet_configs).resolve()
    params["model"]["config_root"] = str(mmdet_configs)


def setup_device(params: Dict[str, Any]) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    params["running"]["device"] = device


def dump_params(params: Dict[str, Any]) -> None:
    with open(PARAMS_FILE, "w") as out_file:
        safe_dump(params, out_file, default_flow_style=False)


def main():
    params = load_params(PARAMS_FILE)
    if params is None:
        raise ValueError('yaml file was not loaded!')

    create_dirs(DIRS)
    install_packages(PACKAGES)
    setup_mmdet(params)
    setup_device(params)
    dump_params(params)
    print("Done!")


if __name__ == "__main__":
    main()
