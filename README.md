## Set up virtual environment
```
python3 -m venv env --system-site-packages
source env/bin/activate
python3 -m pip install -r requirements.txt
```

## Set default pin factory to pigpio
This is done to allow the use of hardware PWM, reducing servo jitter with the GPIO Zero library. \
Add the following lines to the end of `~/.bashrc`
```
export GPIOZERO_PIN_FACTORY=pigpio
```
Start the pigpio daemon, and make it start on boot:
```
sudo pigpiod
sudo systemctl enable pigpiod
```

## Set display size
CatBot's display has been adjusted so that we can see more content. \
Add the following line to the end of `/etc/xdg/lxsession/LXDE-pi/autostart`
```
@xrandr --output DSI-1 --scale 2x2
```

## Auto start on boot
To start CatBot on boot, create `~/.config/autostart/catbot.desktop` with the following content:
```
[Desktop Entry]
Type=Application
Name=CatBot Launcher
Exec=/bin/bash /home/pi/Desktop/CatBot/launcher.sh
```

## Overclocking
To allow the Raspberry Pi to run at 2GHz, change following lines in `/boot/config.txt`
```
over_voltage=6
arm_freq=2000
```

## Other notes:
- CMPS14 settings: See [NOTE-cmps14.md](NOTE-cmps14.md)
