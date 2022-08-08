#
# Copyright (C) 2013 - 2022, Oracle and/or its affiliates. All rights reserved.
# ORACLE PROPRIETARY/CONFIDENTIAL. Use is subject to license terms.
#

"""Initialize environment and local variables during import."""

import os
import tempfile
import atexit
import shutil

from pathlib import Path

JARS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../jars/')
OPG_CLASSPATH = ''

# A list which stores all existing jars.
existing_jars = []
for f in os.listdir(JARS_PATH):
    OPG_CLASSPATH += JARS_PATH + f + ':'
    existing_jars.append(f)
OPG_CLASSPATH = OPG_CLASSPATH[:-1]

if 'OPG_CLASSPATH' in os.environ:
    # Get all paths which are in the classpath variable
    paths = os.environ['OPG_CLASSPATH']
    paths = paths.split(":")

    # Iterate over all paths
    for path in paths:

        # If the path ends with a '*', remove it.
        if path[-1] == "*":
            path = path[:-1]

        # If the path doesn't end with a '/', add it.
        if path[-1] != "/":
            path += "/"

        try:
            # Iterate over all files of this path
            for filename in os.listdir(path):
                # If the file hasn't already been added, add it to OPG_CLASSPATH
                if filename not in existing_jars:
                    OPG_CLASSPATH += ':' + os.path.join(path, filename)
                    existing_jars.append(filename)
        except OSError:
            OPG_CLASSPATH += ':' + path

JAVA_OPTS = []
if 'JAVA_OPTS' in os.environ:
    JAVA_OPTS = os.environ['JAVA_OPTS'].split(' ')

if 'PGX_TMP_DIR' not in os.environ:
    temporary_file = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, path=temporary_file)
    os.environ['PGX_TMP_DIR'] = temporary_file

if 'PGX_GRAPH_ALGORITHM_LANGUAGE' not in os.environ:
    os.environ['PGX_GRAPH_ALGORITHM_LANGUAGE'] = 'JAVA'

if 'PGX_JAVA_HOME_DIR' not in os.environ:
    os.environ['PGX_JAVA_HOME_DIR'] = '<system-java-home-dir>'