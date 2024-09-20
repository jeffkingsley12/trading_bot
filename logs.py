from hmmlearn.base import ConvergenceMonitor

import logging
from binance.lib.utils import config_logging
from logging.handlers import RotatingFileHandler

# Configure logging
logger = logging.getLogger("Bot")
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler(
    "trading_bot.log", maxBytes=5 * 1024 * 1024, backupCount=5
)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d UTC %(levelname)s %(name)s: %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class LoggingMonitor(ConvergenceMonitor):
    def __init__(self, tol, n_iter, verbose=True):
        super().__init__(tol, n_iter, verbose)
        self.iter = 0

    def report(self, loglikelihood):
        if self.verbose:
            logger.info(
                f"HMM Iteration {self.iter}: log-likelihood = {loglikelihood:.2f}"
            )
        self.iter += 1
