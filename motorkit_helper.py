import adafruit_motor.motor
from adafruit_motorkit import MotorKit
from typing import Union, List

kit = MotorKit()

conf_tank = {
    "front_l": 0,
    "front_r": 1,
    "back_l": 2,
    "back_r": 3
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
        motor(target).throttle = speed / 100 

def run_tank(left: float, right: float, steering: float = 0) -> None:
    """
    Run a tank drive at a given speed and steering.

    Args:
        left (float): The speed to run the left motors at (-100 to 100).
        right (float): The speed to run the right motors at (-100 to 100).
        steering (float, optional): The amount to steer (-100 to 100) (default: 0).
    """
    run([conf_tank["front_l"], conf_tank["back_l"]], left + steering)
    run([conf_tank["front_r"], conf_tank["back_r"]], right - steering)

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
