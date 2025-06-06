import os
import glob
import shutil
import subprocess

from setuptools import setup, find_packages


def get_git_info():
      try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                               text=True)
            git_hash = git_hash.strip()
            git_brnc = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                               text=True)
            git_brnc = git_brnc.strip()
            git_stat = subprocess.check_output(['git', 'status', '--porcelain'],
                                                text=True)
            git_stat = 'dirty' if git_stat.strip() else 'clean'
      except (OSError, subprocess.CalledProcessError):
            git_hash = git_brnc = git_stat = 'unknown'
            
      return git_hash, git_brnc, git_stat


def write_version_info():
      ghash, gbrnc, gstat = get_git_info()
      with open(os.path.join('kanarie', 'git_version.py'), 'w') as fh:
            fh.write(f"""
# This is an automatically generated file, do not edit
GIT_BRANCH = '{gbrnc}'
GIT_HASH = '{ghash}'
GIT_STATUS = '{gstat}'
""")


write_version_info()


setup(name                 = "kanarie",
      version              = "0.1.0",
      description          = "Temperature prediction for the LWA",
      long_description     = "Temperature prediction and HVAC monitoring for the LWA shelters",
      author               = "J. Dowell",
      author_email         = "jdowell@unm.edu",
      license              = 'BSD3',
      classifiers          = ['Development Status :: 4 - Beta',
                              'Intended Audience :: Science/Research',
                              'License :: OSI Approved :: BSD License',
                              'Topic :: Scientific/Engineering :: Astronomy'],
      packages             = find_packages(),
      scripts              = glob.glob('scripts/*.py'),
      include_package_data = True,
      python_requires      = ">=3.8",
      install_requires     = ['numpy', 'scikit-learn'],
      zip_safe             = False
)
