import numpy as np
import matplotlib.pyplot as plt
import sys


file = sys.argv[1]
out = sys.argv[2]

data = np.load(file)
plt.figure(figsize=(4,4), dpi=200)
plt.plot(data, label=file)

ymin = 0
if "GE" in file:
    ymax = 255
else:
    ymax = 1

plt.ylim(ymin=ymin,ymax=ymax)
plt.grid(True)
plt.legend()
plt.savefig(out)
