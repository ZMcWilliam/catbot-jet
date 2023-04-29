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
    Returns the motor object for the given number.

    :param num: The motor number (0-3)
    :return: The motor object
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
    Run a motor or a list of motors at a given speed.

    Args:
        targets (Union[int, List[int]]): The motor number(s) (0-3)
        speed (float): The speed to run the motor(s) at (-100 to 100)
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
    Run a tank drive.

    :param left: The speed to run the left motors at (-100 to 100)
    :param right: The speed to run the right motors at (-100 to 100)
    :param steering: (Optional) The amount to steer (-100 to 100) (default: 0)
    """
    run([conf_tank["front_l"], conf_tank["back_l"]], left + steering)
    run([conf_tank["front_r"], conf_tank["back_r"]], right - steering)

def stop(num: int, brake: bool = False) -> None:
    """
    Stop a motor, either coasting or braking.

    :param num: The motor number (0-3)
    :param brake: Whether to brake the motor (True) or just coast (False)
    """
    motor(num).throttle = 0 if brake else None

def stop_all(brake: bool = False) -> None:
    """
    Stop all motors, either coasting or braking.

    :param brake: Whether to brake the motors (True) or just coast (False)
    """
    for i in range(4):
        stop(i, brake)

def stop_bulk(motors: list, brake: bool = False) -> None:
    """
    Stop multiple motors, either coasting or braking.

    :param motors: A list of motor numbers (0-3)
    :param brake: Whether to brake the motors (True) or just coast (False)
    """
    for i in motors:
        stop(i, brake)
