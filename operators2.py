import math
import random
from builtins import round as python_round


class Numbers:
    i = complex(0, 1)


class Matrix:
    def __init__(self, *args):
        for arg in args:
            if not (isinstance(arg, list) or isinstance(arg, tuple)):
                raise TypeError("Arguments for matrix constructor must be lists")
        self.matrix = [list(arg) for arg in args]
        self.columns = len(self.matrix[0])
        self.rows = len(self.matrix)

    def __str__(self):
        return "\n".join([str(row) for row in self.matrix])

    @property
    def flat(self):
        return Matrix(*[[element for row in self.matrix for element in row]])

    def kron(self, other):
        raise NotImplementedError

    def swap(self, index):
        """
        Swaps two adjacent qubits and returns a state vector with appropriate size.
        :param i: the index of the first qubit in the swap, e.g. to swap qubits 2, 3, set i=2
        :return: a new state vector
        """
        ## TODO Untested and won't work until kron is implemeted
        raise NotImplementedError
        assert self.rows == 1 or self.columns == 1
        vec = self.flat
        length = vec.columns
        qubits = int(math.log(length, 2))
        ops = list()
        for _ in range(index):
            ops.append(Matrices.eye(2))
        ops.append(Operators.SWAP)
        while len(ops) < qubits -1:
            ops.append(Matrices.eye(2))
        return (self.kron(ops) if len(ops) > 1 else ops[0]) * vec
        

    @property
    def norm(self):
        """
        Returns the norm of a given vector (row or column)
        :param vector: any vector
        :return: a number
        """
        assert (self.rows == 1 or self.columns == 1)
        return math.sqrt(sum([abs(el**2) for el in self.flat.matrix[0]]))
        
    @property
    def trace(self):
        if self.rows < self.columns:
            return sum([self.matrix[i][i] for i in range(self.rows)])
        else:
            return sum([self.matrix[i][i] for i in range(self.columns)])

    def __add__(self, other):
        if isinstance(other, Matrix):
            m1 = self.matrix
            m2 = other.matrix
            return Matrix(*[[m1[i][j] + m2[i][j] for j in range(self.columns)] for i in range(self.rows)])

    def __sub__(self, other):
        if isinstance(other, Matrix):
            return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(*[[other * self.matrix[i][j] for j in range(self.columns)] for i in range(self.rows)])
        if isinstance(other, Matrix):
            if not (self.rows == other.columns):
                raise ValueError("Multiplied matrices with incompatible dimension")
            return Matrix(*[[sum(x * other.matrix[i][col] for i,x in enumerate(row)) for col in range(len(other.matrix[0]))] for row in self.matrix])

    def __rmul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Matrix(*[[other * self.matrix[i][j] for j in range(self.columns)] for i in range(self.rows)])

    def __getitem__(self, key):
        return self.matrix[key]

    def dot(self, other):
        assert self.rows == 1 or self.columns == 1
        assert other.rows == 1 or other.columns == 1
        v1 = self.flat
        v2 = self.flat.transpose
        return (v1 * v2)[0][0]

    @property
    def transpose(self):
        new_matrix = list()
        for i in range(self.columns):
            new_matrix.append([row[i] for row in self.matrix])
        return Matrix(*new_matrix)

    def __len__(self):
        return len(self.matrix)


class Operators:    
    PauliX = Matrix(
        [0., 1.],
        [1., 0.]
    )
    PauliY = Matrix(
        [0., -Numbers.i],
        [Numbers.i, 0.]
    )
    PauliZ = Matrix(
        [1., 0.],
        [0., -1.]
    )
    PauliI = Matrix(
        [1., 0.],
        [0., 1.]
    )
    Hadamard = Matrix(
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), -1 / math.sqrt(2)]
    )
    CNOT = Matrix(
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.]
    )
    rCNOT = Matrix(
        [1., 0., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.]
    )
    SWAP = Matrix(
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    )
    Toffoli = Matrix(
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    )
    Fredkin = Matrix(
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    )


class Vectors:
    zero = [
        [1],
        [0]
    ]
    zero_zero = [
        [1],
        [0],
        [0],
        [0]
    ]
    one = [
        [0],
        [1]
    ]
    zero_one = [
        [0],
        [1],
        [0],
        [0]
    ]
    two = [
        [0],
        [0],
        [1],
        [0]
    ]
    three = [
        [0],
        [0],
        [0],
        [1]
    ]
    plus = [
        [1 / math.sqrt(2)],
        [1 / math.sqrt(2)]
    ]
    plus_plus = [
        [1 / 2],
        [1 / 2],
        [1 / 2],
        [1 / 2]
    ]
    minus = [
        [1 / math.sqrt(2)],
        [-1 / math.sqrt(2)]
    ]
    minus_minus = [
        [1 / 2],
        [-1 / 2],
        [-1 / 2],
        [1 / 2]
    ]


States = Vectors

class Matrices:
    """
    A collection of standard matrices and vectors, for ease of reference.
    """
    tester1 = [
        [1, 2],
        [3, 4]
    ]
    tester2 = [
        [1, 2, 3],
        [4, 5, 6]
    ]    
    displays = {
        "|1>": Vectors.one,
        "|0>": Vectors.zero,
        "|00>": Vectors.zero_zero,
        "|01>": Vectors.zero_one,
        "|10>": Vectors.two,
        "|11>": Vectors.three,
        "|+>": Vectors.plus,
        "|->": Vectors.minus,
        "|++>": Vectors.plus_plus,
        "|-->": Vectors.minus_minus
    }


    @staticmethod
    def qft(n, precision=10):
        """
        Creates the normalized quantum fourier transform matrix of size n, with entries rounded to precision
        :param n: positive integer
        :param precision: optional, rounding for the entries
        :return: qft matrix of size n
        """
        omega = math.cos(2*math.pi/n) + Numbers.i * math.sin(2*math.pi/2)
        matrix = [[round(omega**(i*j)/math.sqrt(n), precision) for j in range(n)] for i in range(n)]
        return Matrix(*matrix)

    @staticmethod
    def phase(angle):
        """
        Gives a 2x2 phase shift matrix for the given angle
        :param angle: angle in radians
        :return: a 2x2 matrix
        """
        a = math.cos(angle) + Numbers.i * math.sin(angle)
        return Matrix(
            [1, 0],
            [0, a]
        )
    
    @staticmethod
    def mean_inversion(size):
        #TODO Untested
        """
        Creates a matrix to compute a unitary inversion about the mean.
        :param size: the size of the inversion matrix
        :return: the inversion matrix
        """
        base_matrix = Matrices.ones(size)
        identity = Matrices.eye(size)
        two_over_n_matrix = (2/size) * base_matrix
        return two_over_n_matrix - identity

    @property
    def inverse(self):
        raise NotImplementedError

    @property
    def minors(self):
        raise NotImplementedError

    @property
    def cofactor_signs(self):
        raise NotImplementedError

    @property
    def determinant(self):
        raise NotImplementedError

    @property
    def is_unitary(self):
        raise NotImplementedError

    @property
    def is_normal(self):
        raise NotImplementedError

    @property
    def adjoint(self):
        raise NotImplementedError

    def row(self, index):
        """
        Get specified row
        """
        raise NotImplementedError

    def column(self, index):
        """
        Get specified column
        """
        raise NotImplementedError

    def rows_(self, rows):
        """
        Get rows specified by list
        """
        raise NotImplementedError

    def columns_(self, columns):
        """
        Get columns specified by list
        """
        raise NotImplementedError

    def submatrix(self, rows, columns):
        """
        Get a submatrix specified by the rows and columns given as lists.
        """
        raise NotImplementedError("fn not tested")
        return self.rows_(rows).columns_(columns)

    def condense(self):
        """
        Condenses a matrix, i.e. uses row operations to set elements [0][i] to zero for all i except i = 0
        :param matrix: matrix
        :return: condensed matrix
        """
        raise NotImplementedError

    @property
    def eigenvalues(self):
        raise NotImplementedError

    @property
    def eigenvectors(self):
        raise NotImplementedError



    @property
    def conjugate(self):
        raise NotImplementedError

    @property
    def normal(self):
        raise NotImplementedError

    def __pow__(self, power):
        raise NotImplementedError

    @staticmethod
    def zeroes(num_rows, num_cols=-1):
        """
        If only rows is provided, returns a matrix of all zeroes of size rows x rows. If cols is provided, returns same
        of size rows x cols.
        :param num_rows: integer (number of rows)
        :param num_cols: optional, integer (number of columns)
        :return: a matrix filled with zeroes
        """
        if num_cols == -1:
            return Matrix(*[[0. for _ in range(num_rows)] for _ in range(num_rows)])
        else:
            return Matrix(*[[0. for _ in range(num_cols)] for _ in range(num_rows)])

    @staticmethod
    def ones(num_rows, num_cols=-1):
        """
        As zeroes(), but a matrix filled with ones.
        :param num_rows: integer (number of rows)
        :param num_cols: optional, integer (number of columns)
        :return: a matrix filled with ones
        """
        if num_cols == -1:
            return Matrix(*[[1. for _ in range(num_rows)] for _ in range(num_rows)])
        else:
            return Matrix(*[[1. for _ in range(num_cols)] for _ in range(num_rows)])

    @staticmethod
    def eye(shape):
        """
        Gives a square eye matrix of size `shape`
        :param shape: a positive integer
        :return: a square eye matrix
        """
        return Matrix(*[[1. if i == j else 0. for i in range(shape)] for j in range(shape)])

    @staticmethod
    def controlled(matrix):
        """
        Creates the matrix which does the operation of a controlled unitary gate
        :param unitary: a 2x2 unitary matrix
        :return: a 4x4 controlled unitary matrix
        """
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return Matrix(
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0.,  a,  b],
            [0., 0.,  c,  d]
        )

Vector = Matrix

def round(number, n):
    if not isinstance(number, complex):
        return python_round(number, n)
    else:
        return round(number.real, n) + round(number.imag, n) * Numbers.i



zeroes = Operators.PauliY
cont = Matrices.controlled(zeroes)
print(zeroes)
print(cont-cont)
print(3*cont*3)

print(zeroes.transpose)

vec = Matrix([1], [1], [1], [1])
print(vec.norm)
print(vec.flat)
print(vec.dot(vec))
print()
print(Matrices.qft(5))