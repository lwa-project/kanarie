import hashlib
import inspect
import sys
import os
from pathlib import Path

try:
    from .git_version import *
except ImportError:
    GIT_BRANCH = GIT_HASH = GIT_STATUS = 'unknown'
from .model import ShelterTemperatureModel

def _get_code_checksum() -> str:
    """Create a checksum of all relevant module files"""
    modules = ['analysis', 'model', 'utils']
    checksums = []
    
    for module_name in modules:
        module_path = os.path.join(os.path.dirname(__file__), module_name+'.py')
        with open(module_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        checksums.append(f"{module_name}:{file_hash}")

    return ",".join(checksums)

CODE_CHECKSUM = _get_code_checksum()
REPO_INFO = f"{GIT_BRANCH}@{GIT_HASH[:7]}-{GIT_STATUS}"
