from math import sqrt, isclose, sin, cos, pi, log
from builtins import round as python_round
from random import choices, random



class Untested(Exception):
    def __init__(self, message="You are trying to use an untested method or function"):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


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

    def __repr__(self):
        return self.matrix.__repr__()

    def __neg__(self):
        return Matrices.zeroes(self.rows, self.columns) - self

    @property
    def conjugate(self):
        new_matrix = Matrices.zeroes(self.rows, self.columns)
        for i, row in enumerate(self.matrix):
            for j, column in enumerate(row):
                new_matrix[i][j] = self.matrix[i][j].conjugate() if isinstance(self.matrix[i][j], complex) else self.matrix[i][j]
        return Matrix(*new_matrix)

    def pvm(self, vector):
        state = self.density_matrix
        return (vector.adjoint * state * vector).matrix[0][0]


    def multi_swap(self, a, b):
        """
        Swaps the positions of the qubits indexed by numbers a and b, with a strictly less than b.

        Example: Swapping qubits 1 and 5.        

        0 --------------------------------------------- 0

        1 ---------------------\ /--------------------- 5
                                x
        2 ---------------\ /---/ \---\ /--------------- 2
                          x           x
        3 ---------\ /---/ \---------/ \---\ /--------- 3
                    x                       x
        4 ---\ /---/ \---------------------/ \---\ /--- 4
              x                                   x
        5 ---/ \---------------------------------/ \--- 1
        
        """

        assert a < b
        i = b
        j = a +1
        new = self

        ## This loop moves qubit b to the position qubit a by successive swaps.
        ## As a result, every qubit it passes, including qubit a, is moved down one place.
        while i > a:
            new = new.swap(i-1)  # Read as swap(i-1, i)
            i -= 1
        
        ## This loop moves qubit a, which has already been moved down one spot, the rest of the way.
        ## Every qubit it passes is moved up one spot, restoring it to its original place.
        ## The final effect is the swapping of qubits a and b.
        while j < b:
            new = new.swap(j)    # Read as swap(j, j+1)
            j += 1
        return new

    @property
    def flat(self):
        return Matrix(*[[element for row in self.matrix for element in row]])

    def __eq__(self, other):
        assert self.rows == other.rows
        assert self.columns == other.columns
        for i in range(self.rows):
            for j in range(self.columns):
                if self.matrix[i][j] != other.matrix[i][j]:
                    return False
        return True

    @property
    def eigenvalues(self):
        raise NotImplementedError

    @property
    def eigenvectors(self):
        raise NotImplementedError

    def swap_rows(self, row1, row2):
        """
        Swaps two rows in a matrix; a common elementary operation
        :param matrix: a matrix
        :param row1: a row number
        :param row2: a row number
        :return: a matrix
        """
        assert row1 < self.rows
        assert row2 < self.rows
        count = range(self.rows)
        return Matrix(*[self.matrix[i] if i not in [row1, row2] else self.matrix[row1] if i == row2 else self.matrix[row2] for i in count])

    def condense(self):
        """
        Condenses a matrix, i.e. uses row operations to set elements [0][i] to zero for all i except i = 0
        :param matrix: matrix
        :return: condensed matrix
        """
        matrix = self.matrix
        new_matrix = [matrix[0]] + [
            [matrix[i][j] - matrix[i][0] / matrix[0][0] * matrix[0][j] for j in range(self.columns)] for i in
            range(1, self.rows)]
        return Matrix(*new_matrix)

    def measure(self):
        density = self.density_matrix
        probabilities = list()
        possibles = [Matrix(*[[1 if index == each else 0 for each in range(density.rows)]]).transpose for index in range(density.rows)]
        for each in possibles:
            prob = self.pvm(each)
            probabilities.append(f"Probability of measuring state |{str(each.matrix.index([1]))}> is {round(prob.real*100,1)}%.")
        return probabilities


    def hadamard_multiply(self, other):
        """
        Computes the element-wise product of two matrices
        :param matrix1: a matrix
        :param matrix2: a matrix
        :return: a matrix
        """
        assert self.rows == other.rows
        assert self.columns == other.columns

        new_matrix = Matrix(*[[0]*self.columns]*self.rows)

        for i in range(self.rows):
            for j in range(self.columns):
                new_matrix.matrix[i][j] = self.matrix[i][j] * other.matrix[i][j]

        return new_matrix

    ## This is used for the Kronecker product until I can find a better option. Not enough ops in python!
    def __mod__(self, other):
        """
        Calculates the Kronecker product of two matrices.

        For two matrices A, B:

        A = [A11, A12],  B = [B11, B12]
            [A21, A22]       [B21, B22]

            A kron B = [A11.B, A12.B] = [A11.B11, A11.B12, A12.B11, A12.B12]
                    [A21.B, A22.B]   [A11.B21, A11.B22, A12.B21, A12.B22]
                                        [A21.B11, A21.B12, A22.B11, A22.B12]
                                        [A21.B21, A21.B22, A22.B21, A22.B22]

        :param other: A matrix of any size.
        :return: A matrix.
        """
        if isinstance(other, Matrix):
            count = range(other.rows)
            return Matrix(*[[num1 * num2 for num1 in elem1 for num2 in other.matrix[row]] for elem1 in self.matrix for row in count])
        

    def swap(self, index):
        """
        Swaps two adjacent qubits and returns a state vector with appropriate size.
        :param i: the index of the first qubit in the swap, e.g. to swap qubits 2, 3, set i=2
        :return: a new state vector
        """
        assert self.rows == 1 or self.columns == 1
        vec = self.flat
        length = vec.columns
        qubits = int(log(length, 2))
        ops = list()
        for _ in range(index):
            ops.append(Matrices.eye(2))
        ops.append(Operators.SWAP)
        while len(ops) < qubits-1:
            ops.append(Matrices.eye(2))
        
        op = ops[0]
        for each in range(1, len(ops)):
            op = op % ops[each]

        return op * vec.transpose

    @property
    def density_matrix(self):
        return  self.adjoint.normal % self.normal
    
    @property
    def states(self):
        raise NotImplementedError

    def choose(self):
        assert self.rows == 1 or self.columns == 1
        probs = [abs(each)**2 for each in self.normal.flat[0]]
        options = self.density_matrix.states
        selection = choices(options, weights=probs)[0]
        return selection


    @property
    def norm(self):
        """
        Returns the norm of a given vector (row or column)
        :param vector: any vector
        :return: a number
        """
        assert (self.rows == 1 or self.columns == 1)
        return sqrt(sum([abs(el**2) for el in self.flat.matrix[0]]))
        
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
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
            return Matrix(*[[other * self.matrix[i][j] for j in range(self.columns)] for i in range(self.rows)])
        if isinstance(other, Matrix):
            if not (self.columns == other.rows):
                raise ValueError("Multiplied matrices with incompatible dimension")
            return Matrix(*[[sum(x * other.matrix[i][col] for i,x in enumerate(row)) for col in range(len(other.matrix[0]))] for row in self.matrix])

    def __rtruediv__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self / (1/other)

    def __truediv__(self, other):
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, complex):
            return self * (1 / other)

    def __rmul__(self, other):

        if isinstance(other, float) or isinstance(other, int):
            return Matrix(*[[other * self.matrix[i][j] for j in range(self.columns)] for i in range(self.rows)])

    def submatrix(self, rows, columns):
        """
        Get a submatrix specified by the rows and columns given as lists.
        """
        ## TODO: Can request two of the same row, which would delete some of the expected column entries. Acceptable?
        return self.rows_(rows).columns_(columns)

    def __getitem__(self, key):
        return self.matrix[key]

    def dot(self, other):
        assert self.rows == 1 or self.columns == 1
        assert other.rows == 1 or other.columns == 1
        v1 = self.flat
        v2 = self.flat.transpose
        return (v1 * v2)[0][0]

    def row(self, index):
        """
        Get specified row
        """
        return Matrix(self.matrix[index])

    @property
    def inverse(self):
        raise NotImplementedError

    @property
    def minors(self):
        raise NotImplementedError

    @property
    def cofactor_signs(self):
        return Matrix(*[[(-1) ** (i + j) for i in range(self.rows)] for j in range(self.rows)])

    @property
    def determinant(self):
        """
        Recursively calculates the determinant of a matrix by cofactor expansion.

        https://www.youtube.com/watch?v=6-nnxrnnZT8
        """
        if self.rows == self.columns == 1:
            return self.matrix[0][0]
        else:
            det = 0
            for index, value in enumerate(self.matrix[0]):
                cofactor_matrix = self.submatrix(range(1,self.rows), [col for col in range(self.columns) if col != index])
                det = det + value*(-1)**(index+2)*cofactor_matrix.determinant
            return det

    @property
    def is_unitary(self):
        return self * self.adjoint == Matrices.eye(self.rows) and self.is_normal

    @property
    def is_normal(self):
        return self*self.adjoint == self.adjoint*self

    @property
    def adjoint(self):
        return self.transpose.conjugate

    def column(self, index):
        """
        Get specified column
        """
        count = range(self.rows)
        return Matrix(*[[self.matrix[i][index]] for i in count])
        
    @property
    def normal(self):
        """
        The normalized version of a row or column vector
        """
        assert self.rows == 1 or self.columns == 1
        n = self.norm
        return 1/n*self

    def rows_(self, rows):
        """
        Get rows specified by list
        """
        return Matrix(*[self.matrix[i] for i in rows])

    def columns_(self, columns):
        """
        Get columns specified by list
        """
        elements = list()
        for row in self.matrix:
            new_row = list()
            for col in columns:
                new_row.append(row[col])
            elements.append(new_row)
        return Matrix(*elements)

    @property
    def transpose(self):
        new_matrix = list()
        for i in range(self.columns):
            new_matrix.append([row[i] for row in self.matrix])
        return Matrix(*new_matrix)

    def __len__(self):
        return len(self.matrix)


class Operators:    
    DeutschConstant = Matrix(
        [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]
    )
    DeutschBalanced = Matrix(
        [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
    )
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
        [1 / sqrt(2), 1 / sqrt(2)],
        [1 / sqrt(2), -1 / sqrt(2)]
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
    zero = Matrix([1], [0])    
    zero_zero = Matrix(
        [1],
        [0],
        [0],
        [0]
    )
    one = Matrix(
        [0],
        [1]
    )
    zero_one = Matrix(
        [0],
        [1],
        [0],
        [0]
    )
    two = Matrix(
        [0],
        [0],
        [1],
        [0]
    )
    three = Matrix(
        [0],
        [0],
        [0],
        [1]
    )
    plus = Matrix(
        [1 / sqrt(2)],
        [1 / sqrt(2)]
    )
    plus_plus = Matrix(
        [1 / 2],
        [1 / 2],
        [1 / 2],
        [1 / 2]
    )
    minus = Matrix(
        [1 / sqrt(2)],
        [-1 / sqrt(2)]
    )
    minus_minus = Matrix(
        [1 / 2],
        [-1 / 2],
        [-1 / 2],
        [1 / 2]
    )


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
    def random(rows, columns=-1, precision=10):
        if columns == -1:
            return Matrix(*[[round(random(), precision) for row in range(rows)] for row in range(rows)])
        else:
            return Matrix(*[[round(random(), precision) for col in range(columns)] for row in range(rows)])



    @staticmethod
    def qft(n, precision=10):
        """
        Creates the normalized quantum fourier transform matrix of size n, with entries rounded to precision
        :param n: positive integer
        :param precision: optional, rounding for the entries
        :return: qft matrix of size n
        """
        omega = cos(2*pi/n) + Numbers.i * sin(2*pi/2)
        matrix = [[round(omega**(i*j)/sqrt(n), precision) for j in range(n)] for i in range(n)]
        return Matrix(*matrix)

    @staticmethod
    def phase(angle):
        """
        Gives a 2x2 phase shift matrix for the given angle
        :param angle: angle in radians
        :return: a 2x2 matrix
        """
        a = cos(angle) + Numbers.i * sin(angle)
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
    def controlled(control_bit, target_bit, operation):
        """
        Creates the matrix which does the operation of a controlled gate,
        controlled by control_bit, targetting target_bit.

        https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu

        CX[1,3] = |0><0| kr I2 kr I2 + |1><1| kr I2 kr X

        :param control_bit: The control bit deciding whether to perform the operation.
        :param target_bit: Which bit should be operated on if the control is 1.
        :param operation: The operation to be conditionally performed on the target bit.
        :return: a 4x4 controlled unitary matrix
        """
        part1_ops = list()
        part2_ops = list()

        for i in range(max(control_bit, target_bit) + 1):
            if i == control_bit:
                part1_ops.append(Vectors.zero % Vectors.zero.transpose)
                part2_ops.append(Vectors.one % Vectors.one.transpose)
            elif i == target_bit:
                part1_ops.append(Matrices.eye(2))
                part2_ops.append(operation)
            else:
                part1_ops.append(Matrices.eye(2))
                part2_ops.append(Matrices.eye(2))
        
        part1_op = part1_ops[0]
        part2_op = part2_ops[0]
        for i in range(1, len(part1_ops)):
            part1_op = part1_op % part1_ops[i]
            part2_op = part2_op % part2_ops[i]
        
        op = part1_op + part2_op
        return op


        ## (Vectors.zero % Vectors.zero.transpose) % Matrices.eye(4) + (Vectors.one % Vectors.one.transpose) % Matrices.eye(2) % Operators.PauliX


Vector = Matrix

def round(number, n):
    if not isinstance(number, complex):
        return python_round(number, n)
    else:
        return round(number.real, n) + round(number.imag, n) * Numbers.i
