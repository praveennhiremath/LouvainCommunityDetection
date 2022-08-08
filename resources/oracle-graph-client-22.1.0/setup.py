#
# Copyright (C) 2013 - 2022, Oracle and/or its affiliates. All rights reserved.
# ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
#
import glob
import os
import shutil
import subprocess
import sys
import warnings

from pathlib import Path
from setuptools import setup
from setuptools.command.install import install

# The following line may be modified by Gradle when a copy of setup.py
# is made for a PyPGX distribution.

SETUP_PATH = os.path.dirname(os.path.realpath(__file__))
LIB_PATH = os.path.join(SETUP_PATH, "lib")
JARS_PATH = os.path.join(SETUP_PATH, "python", "pypgx", "jars")
WHLS_PATH = os.path.join(SETUP_PATH, "python")
CYTHON_GLOB = os.path.join(WHLS_PATH, "Cython-*.zip")
SIX_GLOB = os.path.join(WHLS_PATH, "six-*.zip")
PYJNIUS_GLOB = os.path.join(WHLS_PATH, "pyjnius-*.zip")
OPG4PY_GLOB = os.path.join(WHLS_PATH, "opg4py-*.zip")


def glob_single_file(pathname):
    names = glob.glob(pathname)
    if not names:
        raise FileNotFoundError("no file matching {!r}".format(pathname))
    if len(names) > 1:
        raise OSError("multiple files matching {!r}".format(pathname))
    return names[0]

try:
    for jar_file in os.listdir(LIB_PATH):
        shutil.copy(os.path.join(LIB_PATH, jar_file), JARS_PATH)

    class InstallLocalPackage(install):
        """A class that installs the necessary libraries."""

        def run(self):
            """Install the necessary libraries."""
            install.run(self)

            if not sys.executable:
                warnings.warn("Empty sys.executable, can't install dependencies of PyPGX")
                return

            no_cython_compile_option = "--install-option='--no-cython-compile'"
            local_inst = '--user' if '--user' in sys.argv else ''
            subprocess.check_call("{} -m pip install {} {} {}"
                            .format(sys.executable, no_cython_compile_option, local_inst, glob_single_file(CYTHON_GLOB)),
                            shell=True)
            subprocess.check_call("{} -m pip install {} {}"
                            .format(sys.executable, local_inst, glob_single_file(SIX_GLOB)),
                            shell=True)
            subprocess.check_call("{} -m pip install {} {}"
                            .format(sys.executable, local_inst, glob_single_file(PYJNIUS_GLOB)),
                            shell=True)
            subprocess.check_call("{} -m pip install {} {}"
                            .format(sys.executable, local_inst, glob_single_file(OPG4PY_GLOB)),
                            shell=True)
            if 'pandas' not in sys.modules:
                # Using --find-links to make sure we don't downgrade existing
                # versions of pandas dependencies already installed by user.
                try:
                    subprocess.check_call("{} -m pip install {} {} --no-index --find-links {}"
                                          .format(sys.executable, "pandas", local_inst, WHLS_PATH),
                                          shell=True, stderr=subprocess.DEVNULL)
                except Exception:
                    warnings.warn(
                        "Error installing pandas. pypgx does not ship with pandas support for "
                        "Python 3.6 onwards. You can install pandas manually."
                    )

    setup(name='pypgx',
          python_requires='>=3.6',
          version='22.1.1',
          description='PyPGX',
          url='PGX',
          platforms=['Linux x86_64'],
          license='OTN',
          long_description='PyPGX',
          packages=[
              'pypgx',
              'pypgx.api',
              'pypgx.api.auth',
              'pypgx.api.filters',
              'pypgx.api.frames',
              'pypgx.api.mllib',
              'pypgx.api.redaction',
              'pypgx.pg',
              'pypgx.pg.rdbms',
              'pypgx._utils',
              'pypgx.jars'
          ],
          package_dir={"pypgx": "python/pypgx"},
          package_data={'pypgx.jars': ['*.jar'],
                        'pypgx.resources': ['*']},
          cmdclass={'install': InstallLocalPackage}
          )

finally:
    for jar_file in os.listdir(LIB_PATH):
        os.remove(os.path.join(JARS_PATH, jar_file))
