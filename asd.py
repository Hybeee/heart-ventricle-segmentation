import numpy as np

se_26_connectivity = np.ones((3, 3, 3))
points = np.stack(se_26_connectivity.nonzero()).T
points -= 1
mask = np.any(points != [0, 0, 0], axis=1)
points = points[mask]

print(points)