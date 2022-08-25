import logging
import time
import os

from utils import master_only

initialized_logger = {}


class StringColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class LoggingTool:
    def __init__(self, file_path, verbose):
        self.name = f"{file_path}/result.log"
        self.verbose = verbose
        self.colors = StringColors()
        self.time = time.localtime
        format_str = '%(asctime)s:%(levelname)s:%(message)s'
        os.makedirs(file_path, exist_ok=True)

        self.logger = logging.Logger(file_path)
        file_handler = logging.FileHandler(f"{file_path}/result.log", 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel([logging.WARNING, logging.INFO, logging.DEBUG][verbose])
        self.logger.addHandler(file_handler)
        initialized_logger[file_path] = True
        # self.logger = logging.basicConfig(
        #     filename=f"{file_path}/result.log",
        #     filemode='w',
        #     level=[logging.WARNING, logging.INFO, logging.DEBUG][verbose],
        #     format='%(asctime)s:%(levelname)s:%(message)s',
        # )

    @master_only
    def info(self, string, is_print=True):
        if is_print:
            print(f"{self.time_updater()} INFO:{string}")
        self.logger.info(string)

    @master_only
    def warning(self, string):
        print(f"{self.colors.WARNING}{self.time_updater()} WARNING: {string}{self.colors.ENDC}")
        self.logger.warning(string)

    def time_updater(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', self.time())
