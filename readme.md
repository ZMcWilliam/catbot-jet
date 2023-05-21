## Set up virtual environment
```
python3 -m venv env --system-site-packages
source env/bin/activate
python3 -m pip install -r requirements.txt
```

## Set default pin factory to pigpio
Add the following lines to the end of `~/.bashrc`:
```
export GPIOZERO_PIN_FACTORY=pigpio
```
Start the pigpio daemon:
```
sudo pigpiod
```
Make the daemon start on boot:
```
sudo systemctl enable pigpiod
```


## Other Notes:
- i2c settings: See [NOTE-i2c.md](NOTE-i2c.md)
- CMPS14 settings: See [NOTE-cmps14.md](NOTE-cmps14.md)
