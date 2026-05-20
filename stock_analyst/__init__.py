from stock_analyst.config.settings import settings
from stock_analyst.config.logging_config import setup_logging

setup_logging(settings.log_level)
