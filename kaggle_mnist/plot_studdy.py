import matplotlib.pyplot as plt
import numpy as np
import time


plt.ion()
c = []
for i in range(400):
    c.append(i)
    plt.plot(c, c)
    # i += 1
    plt.pause(1)
plt.show()
plt.ioff()
