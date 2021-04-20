import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

x = np.arange(0, 100)
y = x * 2
z = x ** 2

# plt.show after every plot command

# EXERCISE 1
fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])    Doesn't work outside of jupyter for this size
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('title')


# EXERCISE 2
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax2 = fig.add_axes([.2, .5, .2, .2])
# ax1 = plt.subplot2grid((1, 1), (0, 0))
# ax2 = plt.subplot2grid((5, 5), (1, 1))
ax1.plot(x, y, color="red")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.plot(x, y, color="red")
ax2.set_xlabel('x')
ax2.set_ylabel('y')
plt.show()


# EXERCISE 3
fig = plt.figure()
ax3 = fig.add_subplot(1, 1, 1)
ax3.plot(x, z, color="red")
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 10000)
ax3.set_xlabel('x')
ax3.set_ylabel('z')
ax4 = fig.add_axes([.2, .5, .4, .4])
ax4.plot(x, y, color="red")
ax4.set_xlim(20, 22)
ax4.set_ylim(30, 50)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('zoom')
plt.show()


# EXERCISE 4
# fig = plt.figure()     not necessary?
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 2))
axes[0].plot(x, y, color="blue", ls="--")
axes[1].plot(x, z, color="red", lw=3)
plt.show()
