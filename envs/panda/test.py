from polygon import is_point_inside_polygon
import numpy as np

import matplotlib.pyplot as plt

N = 100

x_min, x_max = -2, 2
y_min, y_max = -2, 2

# (N, 2)
object_pos = np.random.uniform(x_min, x_max, (N, 2))

# (1, 5, 2)
robot_vertices = np.array([[[-1, 0], [1, 0], [1, 1]]])

plt.figure(figsize=(5, 5))
plt.plot(robot_vertices[0, :, 0], robot_vertices[0, :, 1], 'g-')

is_inside = is_point_inside_polygon(object_pos, robot_vertices, 0)

# if is_inside, plot in blue, else in red
plt.scatter(object_pos[is_inside, 0], object_pos[is_inside, 1], c='b')
plt.scatter(object_pos[~is_inside, 0], object_pos[~is_inside, 1], c='r')
plt.show()