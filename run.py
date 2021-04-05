from WeAreEngineers.run import run
import logging

log_level = logging.INFO

logging.getLogger("WeAreEngineers").setLevel(log_level)
run(model_name="neural_network")

# available model names : random_forest, neural_network, neural_network_search
