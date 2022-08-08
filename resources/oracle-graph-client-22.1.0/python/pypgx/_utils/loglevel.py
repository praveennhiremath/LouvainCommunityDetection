#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#

from jnius import autoclass
import logging as log


def setloglevel(loggername: str = "", loglevel: str = "DEBUG") -> None:
    """Set loglevel for PGX.

    If `loglevel` is invalid (see below), it writes this to the python log as an error
    without raising any exceptions.

    :param loggername: Name of the PGX logger. If empty, ROOT logger's level is updated.
    :param loglevel: Level specification. Must be one of
        `"OFF", "FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE", "ALL"`

    :return: None
    """
    logmanager = autoclass("org.apache.logging.log4j.LogManager")
    configurator = autoclass("org.apache.logging.log4j.core.config.Configurator")
    level = autoclass("org.apache.logging.log4j.Level")

    logger = None
    if (loggername == "") or (loggername.lower() == logmanager.ROOT_LOGGER_NAME.lower()):
        logger = logmanager.getRootLogger()
    else:
        logger = logmanager.getLogger(loggername)

    actuallevel = level.toLevel(loglevel)
    if (actuallevel.toString().lower() != loglevel.lower()):
        log.error("Invalid loglevel specified: {}".format(loglevel))
    else:
        configurator.setLevel(logger.getName(), actuallevel)
