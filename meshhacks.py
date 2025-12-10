import numpy as np

def mungmesh(themesh, func, num_iters=4):
    newmesh = themesh.copy()
    for _ in range(num_iters):
        newmesh = newmesh.subdivide()
    newmesh.vertices = np.apply_along_axis(func, axis=1, arr = newmesh.vertices)
    return newmesh