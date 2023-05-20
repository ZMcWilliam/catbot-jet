The CMPS14 (Compass Module) uses the default address of 0x60, which can conflict with the Adafruit MotorShield. \
To avoid this conflict, I have changed the address of the CMPS14 module to 0x61. 

Follow the steps below to make the necessary changes again:

## Step 1: Install i2c-tools

```shell
sudo apt-get install i2c-tools
```

## Step 2: Connect the CMPS14 Module

Connect the CMPS14 module to the Raspberry Pi, ensure there are no other i2c devices connected.

## Step 3: Identify the Default Address

Once `i2c-tools` is installed, you can use the `i2cdetect` command to identify the address of the CMPS14 module and verify its default address (0x60). Run the following command in the terminal:

```shell
sudo i2cdetect -y 1
```

This command scans the I2C bus with number 1 (you may need to change this number depending on your system configuration) and displays a grid with the detected addresses of the connected devices. Look for the entry corresponding to the CMPS14 module (0x60 by default).

## Step 4: Change the Address

To change the address of the CMPS14 module, you need to write a sequence of commands in the correct order as per https://robot-electronics.co.uk/files/cmps14.pdf. This can be done using the `i2cset` command. Run the following command in the terminal:

```shell
sudo i2cset -y 1 0x60 0x00 0xA0
sudo i2cset -y 1 0x60 0x00 0xAA
sudo i2cset -y 1 0x60 0x00 0xA5
sudo i2cset -y 1 0x60 0x00 0xC2
```

The last command sets the actual address, this can be changed to any value between 0xC0 and 0xCE, I have chosen 0xC2 which correlates to 0x61 in the 7-bit address format.

## Step 5: Verify the New Address

Run the following command in the terminal:

```shell
sudo i2cdetect -y 1
```

You should now see the CMPS14 module at the new address (0x61).

