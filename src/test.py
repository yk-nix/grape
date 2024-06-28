import numpy as np
import matplotlib.pyplot as plt
import time
import socket

from lib.uitls.file import join_path_file

weight_root = 'F:/weight'
print(join_path_file(weight_root, 'lenet5', file = ''))


# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# # s.bind(('0.0.0.0', 2412))
# # while True:
# #   data, _ = s.recvfrom(2048)
# #   print(data.decode())
# count = 0
# while True:
#   count += 1
#   s.sendto(('hello, cout = ' + str(count)).encode(), ('192.168.3.241', 2412))
#   time.sleep(1)
#   print(count)

# fig = plt.figure()
# ax = fig.subplots()

# t = np.linspace(0, 10, 100)
# y = np.sin(t)
# ax.axis([0, 10, 0, 2])
# ax.set_aspect(3)

# while True:
#   ax.plot(t, y)
#   plt.pause(0.5)
#   ax.cla()
#   t += np.pi / 30
#   y = np.sin(t)
