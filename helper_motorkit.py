import adafruit_motor.motor
import time
from adafruit_motorkit import MotorKit
from typing import Union, List

kit = MotorKit()

# Physical port numbers for each motor
conf_tank = {
    "front_l": 1,
    "front_r": 0,
    "back_l": 2,
    "back_r": 3
}

conf_directions = [1, -1, -1, -1] # Motor directions so that a positive speed will move the robot forward
conf_rotation = [50, 50, 56, 47] # The speed for each motor where 1 second correlates to 1 rotation
conf_wheel_size = [60, 60, 50, 50] # Wheel sizes in mm

def normalise_speed(target: int, input_speed: float) -> float:
    """
    Normalise the speed of a motor taking into account its wheel size and rotations per
    second at a baseline speed. This ensures that a given input speed results in the
    same output velocity regardless of the motor's rotation rate or wheel size.

    There will still be some variation in the output speed, but this is significantly
    better than not normalizing the speed at all.

    Args:
        target (int): The motor number (0-3).
        input_speed (float): The desired input speed (-100 to 100).

    Returns:
        float: The normalised speed for the given motor.
    """
    # Calculate the base motor speed factor (for motor with conf_wheel_size of 60)
    base_motor_speed = 50  # This is the speed input that gives 1 rotation per second for the base motor
    base_wheel_size = 60   # This is the wheel size for the base configuration

    # Calculate the current motor's speed factor
    current_motor_factor = conf_rotation[target] / base_motor_speed

    # Adjust for wheel size difference
    current_wheel_factor = conf_wheel_size[target] / base_wheel_size

    # Adjusted speed based on motor factor and wheel factor
    normalised_speed = (input_speed * current_motor_factor) / current_wheel_factor

    return normalised_speed

def motor(num: int) -> adafruit_motor.motor.DCMotor:
    """
    Returns the motor object for a given number.

    Args:
        num (int): The motor number (0-3).

    Returns:
        adafruit_motor.motor.DCMotor: The motor object.
    """
    if num == 0:
        return kit.motor1
    elif num == 1:
        return kit.motor2
    elif num == 2:
        return kit.motor3
    elif num == 3:
        return kit.motor4
    else:
        raise ValueError("Motor number must be between 0 and 3")

def throttle(target: int, speed: float) -> None:
    """
    Set the throttle for a given motor. Skips speed normalisation.

    Args:
        target (int): The motor number (0-3).
        speed (float): The speed to set the motor to (-100 to 100).
    """
    if speed > 100: speed = 100
    elif speed < -100: speed = -100

    motor(target).throttle = speed / 100 * conf_directions[target]

def run(targets: Union[int, List[int]], speed: float) -> None:
    """
    Run one motor or multiple motors at a given speed.

    Args:
        targets (Union[int, List[int]]): The motor number(s) (0-3).
        speed (float): The speed to run the motor(s) at (-100 to 100).
    """

    if isinstance(targets, int):
        targets = [targets]

    if speed > 100: speed = 100
    elif speed < -100: speed = -100

    for target in targets:
        adjusted_speed = normalise_speed(target, speed)
        throttle(target, adjusted_speed)

def run_steer(base_speed: int, max_speed: int, offset: float = 0, skip_range: List[int] = [-15, 25], ramp=False) -> List[float]:
    """
    Run a steering drive at a given speed and offset.

    Args:
        base_speed (int): The base speed to run the motors at (0-100).
        max_speed (int): The maximum speed to run the motors at (0-100).
        offset (float, optional): The offset to apply to the motors for steering (default 0)
        skip_range (List[int], optional): The range of speeds to skip when calculating an offset (default [-30, 30]) - Use False to disable

    Returns:
        List[float]: The final left and right speeds of the motors.
    """
    left_speed = base_speed + offset
    right_speed = base_speed - offset

    if skip_range:
        if skip_range[0] < left_speed < 0:
            left_speed = skip_range[0]
        elif skip_range[1] > left_speed >= 0:
            left_speed = skip_range[1]
        if skip_range[0] < right_speed < 0:
            right_speed = skip_range[0]
        elif skip_range[1] > right_speed >= 0:
            right_speed = skip_range[1]

    left_speed = round(max(min(left_speed, max_speed), -max_speed), 2)
    right_speed = round(max(min(right_speed, max_speed), -max_speed), 2)

    if ramp and left_speed < 30:
        left_speed = 40
    if ramp and right_speed < 30:
        right_speed = 40
    run_tank(left_speed, right_speed)

    return [left_speed, right_speed]

def run_tank(left_speed: int, right_speed: int) -> None:
    """
    Run a tank drive at a given speed.

    Args:
        left_speed (int): The speed to run the left motors at (-100 to 100).
        right_speed (int): The speed to run the right motors at (-100 to 100).
    """
    run([conf_tank["front_l"], conf_tank["back_l"]], left_speed)
    run([conf_tank["front_r"], conf_tank["back_r"]], right_speed)

def run_tank_for_time(left_speed: int, right_speed: int, duration: float, brake: bool = True) -> None:
    """
    Run a tank drive at a given speed for a given duration.

    Args:
        left_speed (int): The speed to run the left motors at (-100 to 100).
        right_speed (int): The speed to run the right motors at (-100 to 100).
        duration (float): The duration to run the motors for (in milliseconds).
        brake (bool, optional): Whether to brake the motors (True) or just coast (False).
    """
    run_tank(left_speed, right_speed)
    time.sleep(duration / 1000)
    stop_all(brake)

def stop(targets: Union[int, List[int]], brake: bool = False) -> None:
    """
    Stop one motor or multiple motors, either coasting or braking.

    Args:
        targets (Union[int, List[int]]): The motor number(s) to stop (0-3) or a list of motor numbers.
        brake (bool, optional): Whether to brake the motor(s) (True) or just coast (False).
    """
    if isinstance(targets, int):
        targets = [targets]

    for target in targets:
        motor(target).throttle = 0 if brake else None

def stop_all(brake: bool = True) -> None:
    """
    Stop all motors, either coasting or braking.

    Args:
        brake (bool, optional): Whether to brake the motors (True) or just coast (False).
    """
    stop([0, 1, 2, 3], brake)
