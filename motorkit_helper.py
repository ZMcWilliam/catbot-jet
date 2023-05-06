import adafruit_motor.motor
import time
from adafruit_motorkit import MotorKit
from typing import Union, List

kit = MotorKit()

# Direction config, such that a positive speed value will be "straight" for each motor
conf_directions = [-1, 1, -1, 1]

conf_tank = {
    "front_l": 1,
    "front_r": 0,
    "back_l": 3,
    "back_r": 2
}
 
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

def run(targets: Union[int, List[int]], speed: float) -> None:
    """
    Run one motor or multiple motors at a given speed.

    Args:
        targets (Union[int, List[int]]): The motor number(s) (0-3).
        speed (float): The speed to run the motor(s) at (-100 to 100).
    """

    if isinstance(targets, int):
        targets = [targets]

    if speed > 100:
        speed = 100
    elif speed < -100:
        speed = -100

    for target in targets:
        motor(target).throttle = speed / 100 * conf_directions[target]

def run_steer(base_speed: int, max_speed: int, offset: float = 0, skip_range: List[int] = [-15, 25]) -> List[float]:
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

def run_tank_for_time(left_speed: int, right_speed: int, duration: float) -> None:
    """
    Run a tank drive at a given speed for a given duration.

    Args:
        left_speed (int): The speed to run the left motors at (-100 to 100).
        right_speed (int): The speed to run the right motors at (-100 to 100).
        duration (float): The duration to run the motors for (in milliseconds).
    """
    run_tank(left_speed, right_speed)
    time.sleep(duration / 1000)
    stop_all()

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
