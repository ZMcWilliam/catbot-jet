import helper_motorkit as m
import time

# m.throttle(0, 50)
# m.throttle(1, 50)
# m.throttle(2, 56)
# m.throttle(3, 47)

m.run_tank(50, 50)

time.sleep(0.5)

m.stop_all()
