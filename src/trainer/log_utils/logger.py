
from logging import Logger, getLogger, FileHandler, StreamHandler, Formatter
import logging

def get_logger(name: str = 'ml-sandbox') -> Logger:
    """Create a logging object in the debugging format.

    Args:
        name (str, optional): Name of the object. Defaults to 'ml-sandbox'.

    Returns:
        Logger: An instance of a logger.
    
    Example:
        >>> # Use the logger
        >>> logger.debug('This is a debug message')
        >>> logger.info('This is an info message')
        >>> logger.error('This is an error message')
    """
    # Create logger
    logger = getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)

    # # Create file handler
    # file_handler = FileHandler('app.log')
    # file_handler.setLevel(logging.DEBUG)

    # Create formatter
    formatter = Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    # logger.addHandler(file_handler)

    return logger