```
                          ██████╗ █████╗ ████████╗██████╗  ██████╗ ████████╗       ██╗███████╗████████╗
_._     _,-'""`-._       ██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗╚══██╔══╝       ██║██╔════╝╚══██╔══╝
(,-.`._,'(       |\`-/|  ██║     ███████║   ██║   ██████╔╝██║   ██║   ██║          ██║█████╗     ██║
    `-.-' \ )-`( , o o)  ██║     ██╔══██║   ██║   ██╔══██╗██║   ██║   ██║     ██   ██║██╔══╝     ██║
        `-    \`_`"'-    ╚██████╗██║  ██║   ██║   ██████╔╝╚██████╔╝   ██║     ╚█████╔╝███████╗   ██║
                          ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝      ╚════╝ ╚══════╝   ╚═╝
```

## CatBot JET 
- RoboCup Junior Rescue Line 2023 - Asia Pacific (South Korea)
- RoboCup Junior Rescue Line 2024 - Singapore Open

![CatBot JET Poster](<./Technical Docs/CatBot JET Poster.jpg>)

## Hardware
| Qty | Item                 | Description                            |
|-----|----------------------|----------------------------------------|
| 1   | Controller     | NVIDIA Jetson Orin Nano                      |
| 1   | Camera         | Raspberry Pi Camera V2 + Fish Eye Lens       |
| 2   | Wheels (back)  | 57A 50mm Omni Wheels                         |
| 4   | Wheels (front) | Pololu 1420 - 60x8mm Wheels                  |
| 4   | Motors         | Pololu 380:1 HPCB 12V Micro Metal Gearmotors |
| 1   | Battery        | Turnigy 2200mAh 3-Cell LiPo Battery          |
| 1   | ADA2348        | Adafruit DC/Stepper Motor Hat                |
| 1   | PCA9685        | Adafruit I2C PWM/Servo Driver                | 
| 1   | U3V70F12       | 12v Step Up Voltage Regulator                |
| 1   | CMPS14         | Magnetic Compass                             |
| 1   | VL53L1X        | Time of Flight Distance Sensor               |
| 1   | RC050S         | 5-inch HDMI Display                          |
| 1   | LM2596S        | 4V-35V Step-Down Converter                   |
| 2   | FS90MG         | Small Servos for Camera and Gate             |
| 1   | FS5115M        | 180deg Large Servo for Claw                  |
| 1   | DSS-M15S       | 270deg Large Servo for Lift                  |

## Software
- Ubuntu 20.04 with NVIDIA JetPack 5.1.2 Installed to a SSD via [SDK-Manager](https://docs.nvidia.com/sdk-manager/index.html)
- Python 3.8
- CUDA 11.4
- cuDNN 8.6
- TensorRT 8.5.2
- TensorFlow 2.12.0
- YOLOv8
- OpenCV 4.5.4 built with CUDA and GStreamer
- Adafruit CircuitPython: MotorKit, ServoKit, PCA9685, VL53l1X

## Technical Docs
You can access the technical docs for CatBot from the links below
> Note: The Engineering Journal and Technical Description Paper are currently only available for CatBot NEO (Made for RoboCup Boardeaux 2023) 

- Poster: **[CatBot JET]()** | [CatBot NEO](https://github.com/ZMcWilliam/catbot-neo/blob/master/Technical%20Docs/CatBot%20NEO%20Poster.pdf)
- Engineering Journal: [CatBot NEO](https://github.com/ZMcWilliam/catbot-neo/blob/master/Technical%20Docs/CatBot%20NEO%20Engineering%20Journal.pdf)
- Technical Description Paper: [CatBot NEO](https://github.com/ZMcWilliam/catbot-neo/blob/master/Technical%20Docs/CatBot%20NEO%20Technical%20Description%20Paper%20Final.pdf)

## Auto start on boot
To start CatBot on boot, create `~/.config/autostart/catbot.desktop` with the following content:
```
[Desktop Entry]
Type=Application
Name=CatBot Launcher
Exec=gnome-terminal -- bash -c "/path/to/catbot/start.sh; exec bash"
```

## Other notes:
- CMPS14 settings: See [NOTE-cmps14.md](NOTE-cmps14.md)
