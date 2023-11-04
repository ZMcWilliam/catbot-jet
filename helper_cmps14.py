import time
import smbus2

class CMPS14:
    """
    Class for interfacing with CMPS14 module over I2C.
    """

    def __init__(self, i2c_bus: int = 1, i2c_address: int = 0x61) -> None:
        """
        Initialize the class with the I2C bus number and the I2C address of the CMPS14 module.

        Args:
            i2c_bus (int, optional): I2C bus number. Defaults to 1.
            i2c_address (int, optional): I2C address of CMPS14 module. Defaults to 0x61.
        """
        self.bus = smbus2.SMBus(i2c_bus)
        self.address = i2c_address

        self.last_values = {
            "bearing_8bit": 0,
            "bearing_16bit": 0,
            "pitch": 0,
            "roll": 0,
        }

    def read_byte(self, register: int) -> int:
        """
        Read a single byte from a specific register.

        Args:
            register (int): Register address.

        Returns:
            int: Byte value read from the register.
        """
        return self.bus.read_byte_data(self.address, register)

    def read_word(self, register: int) -> int:
        """
        Read two bytes from a specific register.

        Args:
            register (int): Register address.

        Returns:
            int: Two byte value read from the register.
        """
        high = self.bus.read_byte_data(self.address, register)
        low = self.bus.read_byte_data(self.address, register + 1)
        return (high << 8) + low

    def write_byte(self, register: int, value: int) -> None:
        """
        Write a single byte to a specific register.

        Args:
            register (int): Register address.
            value (int): Byte value to write.
        """
        self.bus.write_byte_data(self.address, register, value)

    def send_command(self, *commands: int) -> None:
        """
        Send a sequence of bytes to the device, with a 20ms delay between each byte.
        Each byte represents a command to the device.

        Args:
            *commands (int): Sequence of command bytes.
        """
        for cmd in commands:
            self.write_byte(0x00, cmd)
            time.sleep(0.02)  # 20ms delay between each command

    def read_bearing_8bit(self) -> int:
        """
        Reads the compass bearing as a 0-255 value.

        Returns:
            int: Compass bearing in 8-bit.
        """
        try:
            self.last_values["bearing_8bit"] = self.read_byte(0x01)
        except OSError:
            print("[WARN] OSError while reading bearing_8bit. Returning last value.")
        return self.last_values["bearing_8bit"]

    def read_bearing_16bit(self) -> float:
        """
        Reads the compass bearing to form a 16-bit unsigned integer in the range 0-3599
        to represent the bearing (yaw angle). The result is divided by 10 to scale it to 0-359.9°

        Returns:
            float: Compass bearing in 16-bit, scaled to 0-359.9 degrees.
        """
        try:
            value = self.read_word(0x02)
            self.last_values["bearing_16bit"] = value / 10.0 # Scale to 0-359.9°
        except OSError:
            print("[WARN] OSError while reading bearing_16bit. Returning last value.")
        return self.last_values["bearing_16bit"]

    def read_pitch(self) -> int:
        """
        Reads the pitch angle in degrees from the horizontal plane

        Returns:
            int: Pitch angle. (+/- 90°)
        """
        try:
            self.last_values["pitch"] = self.read_byte(0x04)
        except OSError:
            print("[WARN] OSError while reading pitch. Returning last value.")
        return self.last_values["pitch"]

    def read_roll(self) -> int:
        """
        Reads the roll angle in degrees from the horizontal plane

        Returns:
            int: Roll angle. (+/- 90°)
        """
        try:
            self.last_values["roll"] = self.read_byte(0x05)
        except OSError:
            print("[WARN] OSError while reading roll. Returning last value.")
        return self.last_values["roll"]
