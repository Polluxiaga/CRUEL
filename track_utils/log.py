import logging


def get_logger(name='root'):
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger

