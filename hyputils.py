import numpy as np

eps = 1e-6
def klein_to_poincare(vec):
    "how to convert Klein coordinates to Poincare coordinates"

    n2 = np.linalg.norm(vec)**2


    if n2>1+eps:
        raise ValueError("input vector is not in the unit disk")
    elif n2>1:
        n2 = 1

    return vec / (1 + np.sqrt(1 - n2))

def poincare_to_klein(vec):
    "how to convert Poincare coordinates to Klein coordinates"
    n2 = np.linalg.norm(vec)**2

    if n2>1+eps:
        raise ValueError("input vector is not in the unit disk")
    return 2 * vec/(1+n2)

def hyperboloid_to_poincare(vec):
    t = vec[0]
    x = vec[1:]
    return x/(1+t)


def poincare_to_hyperboloid(vec):
    n2 = np.linalg.norm(vec)**2
    if n2>1+eps:
        raise ValueError("input vector is not in the unit disk")
    elif n2>1:
        n2 = 1
    pret = 1+n2
    prex = 2 * vec
    newvec = np.append(pret,prex)/(1-n2)
    return newvec

