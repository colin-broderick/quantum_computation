import matplotlib.pyplot as plt
from operators import Matrices, Vectors, Operators
import time

zero = Vectors.zero % Vectors.zero.transpose
one = Vectors.one % Vectors.one.transpose
I = Matrices.eye(2)
U = Operators.PauliX

a = Vectors.zero % Vectors.zero % Vectors.zero % Vectors.zero
b = Vectors.zero % Vectors.one % Vectors.zero % Vectors.zero
c = Vectors.one % Vectors.zero % Vectors.zero % Vectors.zero
d = Vectors.one % Vectors.one % Vectors.zero % Vectors.zero

e = Vectors.one % Vectors.one % Vectors.zero % Vectors.one

print(zero)
print(one)
print(I)
print(U)

print()

print("zerozerozero", (zero*Vectors.zero).transpose, zero.transpose)

cc124 = zero % zero % I % I  + zero % one % I % I  + one % zero % I % I  + one % one % I % U 
print(a.transpose)
print((cc124 * a).transpose, a._tranpose)
print((cc124 * b).transpose, b.transpose)
print((cc124 * c).transpose, c.transpose)
print((cc124 * d).transpose, d.transpose)
print()
print((a+b+c+e).transpose)

print()
print(one*Vectors.zero)