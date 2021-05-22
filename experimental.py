# import matplotlib.pyplot as plt
# from operators import Matrices, Vectors, Operators, Matrix
# from qmath import rayleigh_quotient
# import time

# # zero = Vectors.zero % Vectors.zero.transpose
# # one = Vectors.one % Vectors.one.transpose
# # I = Matrices.eye(2)
# # U = Operators.PauliX

# # a = Vectors.zero % Vectors.zero % Vectors.zero % Vectors.zero
# # b = Vectors.zero % Vectors.one % Vectors.zero % Vectors.zero
# # c = Vectors.one % Vectors.zero % Vectors.zero % Vectors.zero
# # d = Vectors.one % Vectors.one % Vectors.zero % Vectors.zero

# # e = Vectors.one % Vectors.one % Vectors.zero % Vectors.one

# # print(zero)
# # print(one)
# # print(I)
# # print(U)

# # print()

# # print("zerozerozero", (zero*Vectors.zero).transpose, zero.transpose)

# # cc124 = zero % zero % I % I  + zero % one % I % I  + one % zero % I % I  + one % one % I % U 
# # print(a.transpose)
# # print((cc124 * a).transpose, a.transpose)
# # print((cc124 * b).transpose, b.transpose)
# # print((cc124 * c).transpose, c.transpose)
# # print((cc124 * d).transpose, d.transpose)
# # print()
# # print((a+b+c+e).transpose)

# # print()
# # print(one*Vectors.zero)

# # def error():
# #     ## We can simulate an error producing operator by writing:
# #     ## U = e0 * I + ex * X + ey * Y + ez * Z
# #     ## where e0 is presumably large compared to the others, and the others
# #     ## represent the amplitude to induce their associated error in the measurement.
# #     I = Matrices.eye(2)
# #     X = Operators.PauliX
# #     Y = Operators.PauliY
# #     Z = Operators.PauliZ

# #     zero = Vectors.zero

# #     U = (5*I + 1*X )
# #     print((zero.normal).measure())
# #     print(((U*zero).normal).measure())


# # error()

# # I = Matrices.eye(2)
# # S = Operators.SWAP

# # U = S % I % I * (I%S%I * (I%I%S * (S%I%I * (I%S%I * (I%I%S)))))
# # one = Vectors.zero % Vectors.zero % Vectors.zero % Vectors.one
# # two = Vectors.zero % Vectors.zero % Vectors.one % Vectors.zero
# # three = Vectors.zero % Vectors.zero % Vectors.one % Vectors.one
# # four = Vectors.zero % Vectors.one % Vectors.zero % Vectors.zero
# # print(one.measure())
# # print(two.measure())
# # print(three.measure())
# # print(four.measure())
# # print(U.is_unitary)


# # print((U*two).measure())

# m = Matrix([1,2,3,4],[3,4,5,6],[5,6,7,8],[7,8,9,10.1])
# # m = Matrix([1,0],[0,-1])
# b = Matrix([1.03564321],[2.354654])

# # X=list()
# # Y=list()

# # for i in range(100):
# #     c = (m*b).normal
# #     X.append(c[0][0])
# #     Y.append(c[1][0])
# #     diff = c-b
# #     if abs(diff[0][0]) < 1e-9 and abs(diff[1][0]) < 1e-9:
# #         b = c
# #         print(f"Converged after {i} iterations")
# #         break
# #     b = c
# #     if i == 99:
# #         print("Did not converge")

# # n = [str(i+1) for i in range(len(X))]    

# # print(b*(1/b[1][0]))



# # plt.scatter(X, Y)

# # for i, txt in enumerate(n):
# #     plt.annotate(txt, (X[i], Y[i]))
# # plt.show()

# # v = m.eigenvectors()



# # l = rayleigh_quotient(m, v)

# # print(v)
# # print(l)

# m = Matrix([1,2],[3,4])
# v = Matrix([1],[1])    def __mul__(self, other):
# print((m*v).normal)


import time
from operators import Matrices, Matrix

def mul1(self, other):
    if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
        return Matrix(*[[other * self[i][j] for j in range(self.columns)] for i in range(self.rows)])
    if isinstance(other, Matrix):
        if not (self.columns == other.rows):
            raise ValueError("Multiplied matrices with incompatible dimension")
        return Matrix(*[[sum(x * other[i][col] for i,x in enumerate(row)) for col in range(len(other[0]))] for row in self.matrix])

def dot(one, two):
    return sum([one[0][i]*two[i][0] for i in range(len(two))])

def mul2(self, other):
    newm = Matrices.zeroes(self.rows, other.columns)
    for i in range(newm.rows):
        for j in range(newm.columns):
            newm[i][j] = dot(self.row(i), other.column(j))
    return newm

import numpy as np
from qmath import rayleigh_quotient

def mul3(self, other):
    return Matrix(*(np.matmul(self, other).tolist()))

m1 = Matrix([1,2,3],[4,5,6],[7,8,9])

eigenvectors = m1.eigenvectors()

l = rayleigh_quotient(m1, eigenvectors)
print(l)
print(eigenvectors)

m2 = m1 - 7*Matrices.eye(3)
g = m2.eigenvectors()