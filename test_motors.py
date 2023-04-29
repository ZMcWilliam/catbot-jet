import motorkit_helper as m
import time

m.run_tank(-50, 50)
time.sleep(4)
m.stop_all()
