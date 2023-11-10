from helper_servokit import ServoManager

s = ServoManager()

valid_servos = s.servos.keys()

while True:
    servo_key = input(f"Select servo ({', '.join(valid_servos)}): ")
    if servo_key not in valid_servos:
        raise Exception(f"Invalid servo {servo_key}")

    servo = s.servos[servo_key]

    current_angle = int(servo.angle)
    current_dir = 1
    while True:
        step = input(f"At {current_angle}, enter step (q to quit): ")
        if step == "q":
            break

        if step.startswith("+"):
            try:
                step = int(step[1:])
            except ValueError:
                step = 1
            current_angle += step
            if current_angle > servo.r_max:
                current_angle = servo.r_max
            if current_angle < servo.r_min:
                current_angle = servo.r_min
        else:
            try:
                current_angle = int(step)
            except ValueError:
                current_angle += current_dir

                if current_angle > servo.r_max:
                    current_dir = -1
                    current_angle = servo.r_max - (current_angle - servo.r_max)
                elif current_angle < servo.r_min:
                    current_dir = 1
                    current_angle = servo.r_min + (servo.r_min - current_angle)

        servo.to(current_angle)
