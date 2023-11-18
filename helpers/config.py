import json
import numpy as np

with open("calibration.json", "r", encoding="utf-8") as json_file:
    calibration_data = json.load(json_file)
calibration_map = 255 / np.array(calibration_data["calibration_map_w"])
calibration_map_rescue = 255 / np.array(calibration_data["calibration_map_rescue_w"])

with open("config.json", "r", encoding="utf-8") as json_file:
    config_data_raw = json.load(json_file)

config_data = {}

def process_config(config):
    if config["type"] == "int" or config["type"] == "float":
        return config["data"]["val"]
    elif config["type"] == "hsv":
        # For "hsv", we need to create two np arrays based on L-H, L-S, L-V and H-H, H-S, H-V
        return [
            np.array(bound) for bound in [
                [config["data"]["L-H"], config["data"]["L-S"], config["data"]["L-V"]],
                [config["data"]["H-H"], config["data"]["H-S"], config["data"]["H-V"]]
            ]
        ]
    else:
        raise Exception(f"Unknown config type {config['type']}")

# Each section contains configs, process them individually
for section in config_data_raw.values():
    for key, config in section["configs"].items():
        config_data[key] = process_config(config)

config_values = {
    "black_line_threshold": config_data["black_line_threshold"],
    "black_rescue_threshold": config_data["black_rescue_threshold"],
    "green_turn_hsv_threshold": config_data["green_turn_hsv_threshold"],
    "red_hsv_threshold": config_data["red_hsv_threshold"],
    "rescue_block_hsv_threshold": config_data["rescue_block_hsv_threshold"],
    "rescue_circle_minradius_offset": config_data["rescue_circle_minradius_offset"],
    "rescue_binary_gray_scale_multiplier": config_data["rescue_binary_gray_scale_multiplier"],
    "rescue_circle_conf": {
        "dp": config_data["rescue_circle_dp"],
        "minDist": config_data["rescue_circle_minDist"],
        "param1": config_data["rescue_circle_param1"],
        "param2": config_data["rescue_circle_param2"],
        "minRadius": config_data["rescue_circle_minRadius"],
        "maxRadius": config_data["rescue_circle_maxRadius"],
    },
    "calibration_map": calibration_map,
    "calibration_map_rescue": calibration_map_rescue,
}

def get(key=None):
    """
    Returns the config value for the given key, or all config values if no key is given.

    Args:
        key (str, optional): The key to get the config value for. If None, returns all config values.

    Returns:
        mixed: The config value for the given key, or all config values if no key is given.
    """
    if key is None:
        return config_values
    return config_values[key]

processing_conf = {
    "calibration_map": calibration_map,
    "black_line_threshold": config_values["black_line_threshold"],
    "green_turn_hsv_threshold": config_values["green_turn_hsv_threshold"],
    "red_hsv_threshold": config_values["red_hsv_threshold"],
}
