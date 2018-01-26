# -*- encoding:utf-8 -*-
import logging

# create logger
logger_name = "DAMAGE DETECTION"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

# create file handler
# TODO: optimize the log's name
log_path = "./log.log"
fh = logging.FileHandler(log_path)
fh.setLevel(logging.DEBUG)

# create console handler
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)

# create formatter
fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

# add handler and formatter to logger
fh.setFormatter(formatter)
sh.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(sh)