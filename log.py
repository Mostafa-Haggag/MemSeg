import logging
import logging.handlers


class FormatterNoInfo(logging.Formatter):
    '''
    the formatternoinfo class is  custom log formatter that extends logginer.formatter
    . It modifies how log messages are formatted based on the log level:
    '''
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)
        # The formatter is initialized with a default format string '%(levelname)s: %(message)s'.
        # This format would normally display the log level and the log message (e.g., "INFO: This is a log message").
    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
            # When formatting a log record, it checks the log level (record.levelno).
            # If the log level is INFO, it strips the log message of any additional formatting,
            # returning only the message itself (record.getMessage()).
        # For other log levels (e.g., ERROR, WARNING),
        # it applies the standard formatting as defined by the parent logging.Formatter.
        return logging.Formatter.format(self, record)

'''
The setup_default_logging function configures logging in Python with two main options: console and file logging.
'''
def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    # A StreamHandler is added to output logs to the console.
    console_handler.setFormatter(FormatterNoInfo())
    # It uses a custom formatter (FormatterNoInfo) to format these logs.
    # you must mention the handles and the level of logging
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    # The root logger's level is set to the provided default_level (default is logging.INFO).
    if log_path:
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)