import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

bounds_x = (-1, 1)       # bounds for x
bounds_y = (-1, 1)       # bounds for y
steps = (50, 50)        # number of points
dh = [(b[1] - b[0]) / s for b, s in zip([bounds_x, bounds_y], steps)] # step size
f = lambda x, t: -1     # right-hand function


def closest_border_x(x, y):
    bx = np.sqrt(1 - y ** 2)
    dist_pos = abs(x - bx)
    bx = bx if dist_pos < abs(x + bx) else -bx
    return bx


def closest_border_y(x, y):
    by = np.sqrt(1 - x ** 2)
    dist_pos = abs(y - by)
    by = by if dist_pos < abs(y + by) else -by
    return by

def border_gradient(x, y):
    assert abs(x ** 2 + y ** 2 - 1) < 1e-3
    return 0


def is_inside_region(x, y):
    return x ** 2 + y ** 2 <= (1 - 1e-3)

X = np.linspace(*bounds_x, steps[0])
Y = np.linspace(*bounds_y, steps[1])
X_, Y_ = np.meshgrid(X, Y)

inside_region = is_inside_region(X_, Y_)

#

fig = plt.figure()
ax = plt.axes(projection='3d')


X = np.linspace(*bounds_x, steps[0])
Y = np.linspace(*bounds_y, steps[1])

X, Y = np.meshgrid(X, Y)
Z = is_inside_region(X, Y)


ax.contour3D(X, Y, Z, 50, cmap='binary')
plt.show()

#

equations = []
B = []
points = set()
# border_points = set()
for y in range(inside_region.shape[0]):
    for x in range(inside_region.shape[1]):
        if not inside_region[y, x]:
            continue
        x_, y_ = X[x], Y[y]
        points.add((x_, y_))
        equations.append([])
        # center
        equations[-1].append([-4 * dh[0] * dh[1], (x_, y_)])

        # X points are both inside the region
        if inside_region[y, x - 1] and inside_region[y, x + 1]:
            equations[-1].append([1 / dh[0] ** 2, (X[x - 1], y_)])
            equations[-1].append([1 / dh[0] ** 2, (X[x + 1], y_)])
            points.add((X[x - 1], y_))
            points.add((X[x + 1], y_))
        else:
            if inside_region[y, x - 1]:
                clx = closest_border_x(y_, X[x + 1])
                lmbd = dh[0] / abs(clx - x_)
                #                 print(lmbd)
                #                 equations[-1].append([lmbd / (1+lmbd), (clx, y_)])
                equations[-1].append([1 / (1 + lmbd), (X[x - 1], y_)])
                #                 border_points.add((clx, y_))
                points.add((X[x - 1], y_))
            else:
                clx = closest_border_x(y_, X[x - 1])
                lmbd = dh[0] / abs(clx - x_)
                #                 print(lmbd)
                #                 equations[-1].append([lmbd / (1+lmbd), (clx, y_)])
                equations[-1].append([1 / (1 + lmbd), (X[x + 1], y_)])
                #                 border_points.add((clx, y_))
                points.add((X[x + 1], y_))

        # Y points are both inside the region
        if inside_region[y - 1, x] and inside_region[y + 1, x]:
            equations[-1].append([1 / dh[1] ** 2, (x_, Y[y - 1])])
            equations[-1].append([1 / dh[1] ** 2, (x_, Y[y + 1])])
        else:
            if inside_region[y - 1, x]:
                cly = closest_border_y(x_, Y[y + 1])
                lmbd = dh[1] / abs(cly - y_)
                #                 print(lmbd)
                #                 equations[-1].append([lmbd / (1+lmbd), (x_, cly)])
                equations[-1].append([1 / (1 + lmbd), (x_, Y[y - 1])])
                #                 border_points.add((x_, cly))
                points.add((x_, Y[y - 1]))
            else:
                cly = closest_border_y(x_, Y[y - 1])
                lmbd = dh[1] / abs(cly - y_)
                #                 print(lmbd)
                #                 equations[-1].append([lmbd / (1+lmbd), (x_, cly)])
                equations[-1].append([1 / (1 + lmbd), (x_, Y[y + 1])])
                #                 border_points.add((x_, cly))
                points.add((x_, Y[y + 1]))

        B.append(f(x_, y_))

# points, border_points = list(points), list(border_points)
points = list(points)
mapping = {(x, y): ind for ind, (x, y) in enumerate(points)}
# mapping.update({(x, y): ind+len(points) for ind, (x,y) in enumerate(border_points)})


# assert len(set(mapping.values())) == (len(border_points) + len(points))
assert len(set(mapping.values())) == len(points)

A = np.zeros((len(mapping), len(mapping)), dtype=np.float32)

for l, eq in enumerate(equations):
    for coeff, p in eq:
        A[l][mapping[p]] = coeff

# for l, p in enumerate(border_points):
#     l += len(points)
#     A[l][mapping[p]] = 1
#     B.append(0)

B = np.asarray(B)

assert A.shape[0] == A.shape[1] == B.size

solution = solve(A, B)
solution = solution.reshape(B.size)

fig = plt.figure()
ax = plt.axes(projection='3d')

XY = points
X, Y = zip(*XY)
X, Y = np.asarray(X), np.asarray(Y)
Z = solution

ax.scatter(X, Y, Z, cmap='binary')
plt.show()