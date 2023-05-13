The default frequency of the i2c bus on a Raspberry Pi seemed to make reading from the line array/colour sensors rather slow.
By increasing the speed, an increase from ~15FPS to ~ 45FPS can easily be achieved when running the full program

To do this:

1. Open **/boot/config.txt** file

	`sudo nano /boot/config.txt`
    
2. Find the line containing `dtparam=i2c_arm=on`

3. Add `i2c_arm_baudrate=<new speed>` (Separate with a Comma)

    `dtparam=i2c_arm=on,i2c_arm_baudrate=400000`

4. Reboot Raspberry Pi