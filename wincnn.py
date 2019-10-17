import torch
torch.nn.




from sympy import Matrix, MatrixSymbol, Rational, simplify, cse, pprint, symbols, init_printing
init_printing(use_unicode=True)

AT = Matrix([[1, 1,  1,  0],
             [0, 1, -1, -1]])
A = AT.T

BT = Matrix([[1,  0, -1,  0],
             [0,  1,  1,  0],
             [0, -1,  1,  0],
             [0,  1,  0, -1]])
B = BT.T

half = Rational(1, 2)
G = Matrix([[1,  0,  0],
            [half, half, half],
            [half, -half, half],
            [0,  0,  1]])

g = MatrixSymbol("g", 3, 3)
d = MatrixSymbol("d", 4, 4)

pprint("Filter transformation")
gs = simplify((G @ g @ G.T).as_explicit())
pprint(gs[:, 0])
pprint(gs[:, 1])
pprint(gs[:, 2])


pprint("Data transformation")
ds = simplify((BT @ d @ B).as_explicit())
pprint(ds[:, 0])
pprint(ds[:, 1])
pprint(ds[:, 2])


gs = [MatrixSymbol(f"gs{x}", 4, 4) for x in range(4)]
ds = [MatrixSymbol(f"ds{x}", 4, 4) for x in range(4)]

# variant 1
r = [AT @ (gs[x] * ds[x]) @ A for x in range(4)]
r = r[0] + r[1] + r[2] + r[3]
r_normal = r.as_explicit()

# variant 2
r = [gs[x] * ds[x] for x in range(4)]
r = r[0] + r[1] + r[2] + r[3]
r = AT @ r @ A
r_new = r.as_explicit()

simplify(r_new - r_normal)

import numpy as np
w = np.ones((3, 3)) * 4
d = np.ones((4, 4)) * 4

w = G@w@G.T
d = BT@d@B