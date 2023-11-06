import time
import helper_motorkit as m

# m.motor(0).throttle = 0.5
# m.motor(1).throttle = 0.5
# m.motor(2).throttle = 0.5
# m.motor(3).throttle = 0.5

# m.throttle(0, 50)
# m.throttle(1, 50)
# m.throttle(2, 56)
# m.throttle(3, 47)

m.run_tank(-50, 50)

time.sleep(2)

m.stop_all()
