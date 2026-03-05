
import json
import logging

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

logger = logging.getLogger(__name__)


def write_json(config_filename, data):

    with open(config_filename, 'w') as config_file:
        json.dump(data, config_file)

    logger.info(f"Data successfully written to {config_filename}")


def read_json(config_filename):

    with open(config_filename, 'r') as config_file:
        data_loaded = json.load(config_file)

    logger.info(f"Data loaded from {config_filename}")

    return data_loaded