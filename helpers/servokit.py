from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

class Servo:
    def __init__(self, kit: ServoKit, port, actuation_range=180, min_pulse_width=1000, max_pulse_width=2000, initial_angle=0, r_min=0, r_max=180):
        self.kit = kit
        self.port = port
        self.actuation_range = actuation_range
        self.r_min = r_min
        self.r_max = r_max
        self.angle = initial_angle

        self.kit.servo[self.port].actuation_range = actuation_range
        self.kit.servo[self.port].set_pulse_width_range(min_pulse_width, max_pulse_width)
        self.kit.servo[self.port].angle = self.angle

    def to(self, angle):
        if 0 <= angle <= self.actuation_range:
            self.angle = angle
            self.kit.servo[self.port].angle = self.angle
        else:
            raise ValueError(f"Angle {angle} is out of range (0-{self.actuation_range}) for servo {self.port}")

    def toMin(self):
        self.to(self.r_min)

    def toMax(self):
        self.to(self.r_max)

class ServoManager:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.servos = {
            "gate": self.create_servo(0, 180, 500, 2500, 15, 15, 130),
            "lift": self.create_servo(1, 270, 500, 2500, 10, 10, 200),
            "cam":  self.create_servo(2, 180, 500, 2500, 35, 35, 90),
            "claw": self.create_servo(3, 180, 500, 2500, 20, 10, 77),
        }

    def create_servo(self, port, actuation_range, min_pulse_width, max_pulse_width, initial_angle, r_min, r_max):
        servo = Servo(self.kit, port, actuation_range, min_pulse_width, max_pulse_width, initial_angle, r_min, r_max)
        return servo

    def __getattr__(self, item):
        return self.servos[item]
