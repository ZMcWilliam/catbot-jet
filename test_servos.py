from gpiozero import AngularServo

PORT_SERVO_GATE = 12
PORT_SERVO_CLAW = 13
PORT_SERVO_LIFT = 18
PORT_SERVO_CAM = 19

servo = {
    "gate": AngularServo(PORT_SERVO_GATE, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-90),    # -90=Close, 90=Open
    "claw": AngularServo(PORT_SERVO_CLAW, min_pulse_width=0.0005, max_pulse_width=0.002, initial_angle=-80),    # 0=Open, -90=Close
    "lift": AngularServo(PORT_SERVO_LIFT, min_pulse_width=0.0005, max_pulse_width=0.0025, initial_angle=-80),   # -90=Up, 40=Down
    "cam": AngularServo(PORT_SERVO_CAM, min_pulse_width=0.0006, max_pulse_width=0.002, initial_angle=-83)       # -90=Down, 90=Up
}

while True:
    selected_servo = input("Select servo (gate, claw, lift, cam): ")
    if selected_servo not in servo.keys():
        raise Exception(f"Invalid servo {selected_servo}")

    selected_servo = servo[selected_servo]

    current_angle = int(selected_servo.angle)
    current_dir = 1
    while True:
        step = input(f"At {current_angle}, enter step (q to quit): ")
        if step == "q":
            break

        if step.startswith("+"):
            try:
                step = int(step[1:])
            except:
                step = 1
            current_angle += step
            if current_angle > 90:
                current_angle = 90
            if current_angle < -90:
                current_angle = -90
        else:
            try:
                current_angle = int(step)
            except:
                current_angle += current_dir

                if current_angle > 90:
                    current_dir = -1
                    current_angle = 90 - (current_angle - 90)
                elif current_angle < -90:
                    current_dir = 1
                    current_angle = -90 + (-90 - current_angle)

        selected_servo.angle = current_angle