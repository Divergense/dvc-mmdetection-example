import os
import sys
import importlib
import subprocess

from abc import ABCMeta, abstractmethod


class PackageWrapperBase(metaclass=ABCMeta):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def install(self) -> None:
        pass


class PackageWrapperPip(PackageWrapperBase):
    """
    Regular pip installation of an package.
    """
    
    def __init__(self, name: str, version=None):
        super().__init__(name)
        self._version = version

    @property
    def version(self) -> str:
        return self._version

    def install(self) -> None:
        package_name = self.name
        if self.version is not None:
            package_name += f'=={self.version}'
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


class  PackageWrapperGithub(PackageWrapperBase):
    """
    Clone github repo in current directory and
    install them using pip.
    """
    
    def __init__(self, name: str, url: str):
        super().__init__(name)
        self._url = url

    @property
    def url(self) -> str:
        return self._url

    def install(self) -> None:
        package_name = self.name
        github_url = self.url
        subprocess.run(["rm", "-rf", package_name])
        subprocess.run(["git", "clone", github_url])
        os.chdir(package_name)
        subprocess.run(["pip", "install", "-e", "."])
        os.chdir("..")


class PackageWrapperMmcv(PackageWrapperBase):
    base_package = 'openmim'
    base_package_installed = False

    def __init__(self, name: str):
        super().__init__(name)

    def install(self) -> None:
        openmim = PackageWrapperPip(self.base_package)
        if not self.base_package_installed:
            try:
                importlib.import_module(openmim.name)
            except ImportError:
                openmim.install()
                
            self.base_package_installed = True
        
        subprocess.run(["min", "install", self.name])
