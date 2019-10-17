import numpy as np

data = np.zeros((8, 8, 4), dtype=np.uint8)
result = np.zeros((6, 6), dtype=np.int32)
weight = np.zeros((3, 3), dtype=np.int32)
# i = 0
# for y in range(data.shape[0]):
#     for x in range(data.shape[1]):
#         i += 2
#         print(f"d1 = data[{y}][{x}][0];")
#         print(f"d2 = data[{y}][{x}][1];")
#         for kx in range(3):
#             for ky in range(3):
#                 if (x - kx >= 0) and (x-kx < result.shape[1]) and (y-ky >= 0) and (y-ky < result.shape[0]):
#                     w = ky, kx
#                     ry, rx = y-ky, x-kx
#                     # print(f"DATA {d}, WEIGHTS {w}, RESULT {ry, rx}")
#                     print(f"w = threadWeights[{ky}][{kx}][0];")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d1.x, w.x, cache[{ry}][{rx}]);")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d1.y, w.y, cache[{ry}][{rx}]);")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d1.z, w.z, cache[{ry}][{rx}]);")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d1.w, w.w, cache[{ry}][{rx}]);")
#
#                     print(f"w = threadWeights[{ky}][{kx}][0];")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d2.x, w.x, cache[{ry}][{rx}]);")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d2.y, w.y, cache[{ry}][{rx}]);")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d2.z, w.z, cache[{ry}][{rx}]);")
#                     print(f"cache[{ry}][{rx}] = __dp4a(d2.w, w.w, cache[{ry}][{rx}]);")
#                     print()
#                     i += 10
#         print()
#         print()


data = np.zeros((6, 6, 4), dtype=np.uint8)
result = np.zeros((4, 4), dtype=np.int32)
weight = np.zeros((3, 3), dtype=np.int32)
i = 0
l = []
for y in range(1):
    for x in range(6):
        i += 2
        # print(f"d0 = data[{y} + threadIdx.x][{x}][4*0];")
        print(f"d0 = data[y + threadYOffset][{x} + threadXOffset][0];")
        print(f"d1 = data[y + threadYOffset][{x} + threadXOffset][1];")
        print(f"d2 = data[y + threadYOffset][{x} + threadXOffset][2];")
        print(f"d3 = data[y + threadYOffset][{x} + threadXOffset][3];")
        print(f"d4 = data[y + threadYOffset][{x} + threadXOffset][4];")
        print(f"d5 = data[y + threadYOffset][{x} + threadXOffset][5];")
        print(f"d6 = data[y + threadYOffset][{x} + threadXOffset][6];")
        print(f"d7 = data[y + threadYOffset][{x} + threadXOffset][7];")

        for kx in range(3):
            if (x - kx >= 0) and (x - kx < result.shape[1]):
                rx = x - kx
                print(f"w1 = threadWeights[{kx}][0];")
                print(f"w2 = threadWeights[{kx}][1];")
                print(f"cache{rx} = __dp4a(d0, w1.x, cache{rx});")
                print(f"cache{rx} = __dp4a(d1, w1.y, cache{rx});")
                print(f"cache{rx} = __dp4a(d2, w1.w, cache{rx});")
                print(f"cache{rx} = __dp4a(d3, w1.z, cache{rx});")
                print(f"cache{rx} = __dp4a(d4, w2.x, cache{rx});")
                print(f"cache{rx} = __dp4a(d5, w2.y, cache{rx});")
                print(f"cache{rx} = __dp4a(d6, w2.w, cache{rx});")
                print(f"cache{rx} = __dp4a(d7, w2.z, cache{rx});")
                print()
                i += 10
        print()
        print()
        # for kx in range(3):
        #     for ky in range(3):
        #         if (x - kx >= 0) and (x-kx < result.shape[1]) and (y-ky >= 0) and (y-ky < result.shape[0]):
        #             w = ky, kx
        #             ry, rx = y-ky, x-kx
        #             if ry != 0:
        #                 continue
        #             l.append(rx)
        #             print(f"w = threadWeights[{ky}][{kx}];")
        #             print(f"cache{rx} = __dp4a(d0, w.x, cache{rx});")
        #             print(f"cache{rx} = __dp4a(d1, w.y, cache{rx});")
        #             print(f"cache{rx} = __dp4a(d2, w.w, cache{rx});")
        #             print(f"cache{rx} = __dp4a(d3, w.z, cache{rx});")
        #             print()
        #             i += 10
        # print()
        # print()


print("uint32_t dxOffset = 4 + threadIdx.x / 2;")
print("uint32_t dyOffset = (threadIdx.x / 2) * 3;")

for y in range(5):
    for x in range(3):
        i += 2
        # print(f"d11 = data[dyOffset + {y}][dxOffset + {x}][0];")
        # print(f"d12 = data[dyOffset + {y}][dxOffset + {x}][1];")
        # print(f"d21 = data[dyOffset + {y}][dxOffset + {x}][2];")
        # print(f"d22 = data[dyOffset + {y}][dxOffset + {x}][3];")
        print(f"d.x = data[dyOffset + {y}][dxOffset + {x}][2*chn];")
        print(f"d.y = data[dyOffset + {y}][dxOffset + {x}][2*chn+1];")
        for kx in range(3):
            for ky in range(3):
                if (x - kx >= 0) and (x-kx < result.shape[1]) and (y-ky >= 0) and (y-ky < result.shape[0]):
                    w = ky, kx
                    ry, rx = y-ky, x-kx
                    if rx != 0 or ry > 2:
                        continue
                    ry += 6
                    l.append(ry)

                    print(f"w = threadWeights[{ky}][{kx}];")
                    print(f"cache{ry} = __dp4a(d.x, w.x, cache{ry});")
                    print(f"cache{ry} = __dp4a(d.y, w.y, cache{ry});")
                    # print(f"DATA {d}, WEIGHTS {w}, RESULT {ry, rx}")
                    # print(f"w = threadWeights[{ky}][{kx}][0];")
                    # print(f"cache[{ry}] = __dp4a(d11.x, w.x, cache[{ry}]);")
                    # print(f"cache[{ry}] = __dp4a(d11.y, w.y, cache[{ry}]);")
                    # print(f"cache[{ry}] = __dp4a(d12.x, w.z, cache[{ry}]);")
                    # print(f"cache[{ry}] = __dp4a(d12.y, w.w, cache[{ry}]);")
                    #
                    # print(f"w = threadWeights[{ky}][{kx}][1];")
                    # print(f"cache[{ry}] = __dp4a(d21.x, w.x, cache[{ry}]);")
                    # print(f"cache[{ry}] = __dp4a(d21.y, w.y, cache[{ry}]);")
                    # print(f"cache[{ry}] = __dp4a(d22.x, w.z, cache[{ry}]);")
                    # print(f"cache[{ry}] = __dp4a(d22.y, w.w, cache[{ry}]);")
                    # print()
                    i += 10
        print()
        print()

A = np.zeros((2, 8, 8))
for ty in range(32):
    for tx in range(4):
        y = (tx + ty * 4) // 64
        x = ((tx + ty * 4) - y * 64) // 8
        chn = (tx + ty * 4) - y * 64 - x * 8
        A[y][x][chn] += 1
        print(y, x, chn)
# // y = 0, x = 0, 1 element to contribute
# auto& d = data[0][0][tmp];
#
# auto& w = threadWeights[0][0];
# accumulator = __dp4a(d.x, w.x, accumulator);
# accumulator = __dp4a(d.y, w.y, accumulator);
# cache[0][0] += accumulator;
#
# accumulator = 0;

import numpy as np

data = np.zeros((4, 4, 4), dtype=np.uint8)
result = np.zeros((2, 2), dtype=np.int32)
weight = np.zeros((3, 3), dtype=np.int32)

comands = []

for y in range(data.shape[0]):
    for x in range(data.shape[1]):
        d = (y, x)
        com = [f"d1 = data[{y}][{x}][0];", f"d2 = data[{y}][{x}][1];"]
        for kx in range(3):
            for ky in range(3):
                if (x - kx >= 0) and (x-kx < result.shape[1]) and (y-ky >= 0) and (y-ky < result.shape[0]):
                    w = ky, kx
                    ry, rx = y-ky, x-kx
                    com += [
                        f"w = threadWeights[{ky}][{kx}][0];",
                        f"cache[{ry}][{rx}] = __dp4a(d1.x, w.x, cache[{ry}][{rx}]);",
                        f"cache[{ry}][{rx}] = __dp4a(d1.y, w.y, cache[{ry}][{rx}]);",
                        f"cache[{ry}][{rx}] = __dp4a(d1.z, w.z, cache[{ry}][{rx}]);",
                        f"cache[{ry}][{rx}] = __dp4a(d1.w, w.w, cache[{ry}][{rx}]);"
                    ]

                    com += [
                        f"w = threadWeights[{ky}][{kx}][0];",
                        f"cache[{ry}][{rx}] = __dp4a(d2.x, w.x, cache[{ry}][{rx}]);",
                        f"cache[{ry}][{rx}] = __dp4a(d2.y, w.y, cache[{ry}][{rx}]);",
                        f"cache[{ry}][{rx}] = __dp4a(d2.z, w.z, cache[{ry}][{rx}]);",
                        f"cache[{ry}][{rx}] = __dp4a(d2.w, w.w, cache[{ry}][{rx}]);",
                    ]
        comands.append(com)

for x in range(17):
    for y in range(17):
        if (x+2)*(y+2) == 128:
            print(x, y)


import numpy as np
A = np.zeros((10, 10, 4), dtype=np.float32)
B = A.reshape(-1)

for ind in range(3):
    for ty in range(32):
        for tx in range(4):
            g = tx + ty*4

            is_half_warp = (ty // 4) % 2
            g += -2*(is_half_warp * 16) + 16 + ind*128

            chn = int(g % 4)
            y = g // 40
            x = (g - y*40) // 4

            A[y][x][chn] += 1



for ty in range(32):
    for tx in range(8):
        g = tx + ty*8 + 256
        chn = int(g % 8)
        y = g // 80
        x = (g - y*80) // 8

        A[y][x][chn] += 1

for ty in range(32):
    for tx in range(8):
        g = tx + ty*8 + 512
        chn = int(g % 8)
        y = g // 80
        x = (g - y*80) // 8

        A[y][x][chn] += 1










import numpy as np
A = np.zeros((32, 3, 3, 8), dtype=np.float32)
for ind in range(9):
    for ty in range(32):
        for tx in range(8):
            g = tx + ty*8 + 256*ind
            chn_out = g // 72
            y = int((g // 24) % 3)
            x = int((g // 8) % 3)
            chn_in = g % 8

            A[chn_out][y][x][chn_in] += 1








data = np.zeros((10, 10, 4), dtype=np.uint8)
result = np.zeros((8, 8), dtype=np.int32)
weight = np.zeros((3, 3), dtype=np.int32)
# i = 0
i = 0
l = []
for y in range(3):
    for x in range(10):
        i += 1
        print(f"d = data[threadIdx.x + {y}][{x}][chn];")
        for kx in range(3):
            for ky in range(3):
                if (x - kx >= 0) and (x-kx < result.shape[1]) and (y-ky >= 0) and (y-ky < result.shape[0]):
                    w = ky, kx
                    ry, rx = y-ky, x-kx
                    if ry != 0:
                        continue
                    l.append(rx)
                    # print(f"DATA {d}, WEIGHTS {w}, RESULT {ry, rx}")

                    print(f"w = threadWeights[{ky}][{kx}];")
                    print(f"cache{rx} = __dp4a(d, w, cache{rx});")
                    print()
                    i += 3
        print()
        print()



import numpy as np
A = np.zeros((16, 16, 5))
B = A.reshape(-1)

i = 0
for j in range(B.size):
    B[j] = i
    i += 1
    i = i % 32

from collections import Counter

# C = A.reshape(-1, A.shape[-1])
# c = C[:2, :16*4]
c = A[:2, :, 4]
Counter(Counter(c.ravel().tolist()).values())


# C = A[:, :130].reshape(10, 10, 13)
#
# np.unique(C[0, :, :-1:4][:, :2].ravel()).shape
#
C[:2, :, :-1:4][:, :, :2].ravel()[:32]
Counter(Counter(C[:2, :, :-1:4][:, :, :2].ravel()[:32].tolist()).values())

A[:2, :10, :-1:4][:, :, :2].ravel()[:32]
from collections import Counter
Counter(Counter(A[:2, :, :-1:4][:, :, :2].ravel()[:32].tolist()).values())