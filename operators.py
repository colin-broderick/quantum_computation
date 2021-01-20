import math
import random

from builtins import round as python_round

"""
Column vectors are lists of lists:
    e.g. A = [[0], [1], [2], [3]].
Row vectors are a list of a single list of integers:
    e.g. B = [[0, 1, 2, 3]].
Matrices are lists of lists of integers, where each inner list represents a row:
    e.g. C = [[0, 1], [1, 0]] is PauliX.
"""

## converted
class Matrices:
    """
    A collection of standard matrices and vectors, for ease of reference.
    """
    i = complex(0, 1)
    tester1 = [
        [1, 2],
        [3, 4]
    ]
    tester2 = [
        [1, 2, 3],
        [4, 5, 6]
    ]
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
    PauliX = [
        [0, 1],
        [1, 0]
    ]
    PauliY = [
        [0, -i],
        [i, 0]
    ]
    PauliZ = [
        [1, 0],
        [0, -1]
    ]
    PauliI = [
        [1, 0],
        [0, 1]
    ]
    Hadamard = [
        [1 / math.sqrt(2), 1 / math.sqrt(2)],
        [1 / math.sqrt(2), -1 / math.sqrt(2)]
    ]
    CNOT = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ]
    rCNOT = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ]
    SWAP = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]
    Toffoli = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ]
    Fredkin = [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ]
    displays = {
        "|1>": one,
        "|0>": zero,
        "|00>": zero_zero,
        "|01>": zero_one,
        "|10>": two,
        "|11>": three,
        "|+>": plus,
        "|->": minus,
        "|++>": plus_plus,
        "|-->": minus_minus
    }


## CONVERTED
def phase(angle):
    """
    Gives a 2x2 phase shift matrix for the given angle
    :param angle: angle in radians
    :return: a 2x2 matrix
    """
    a = math.cos(angle) + Matrices.i * math.sin(angle)
    return [
        [1, 0],
        [0, a]
    ]

### CONVERTED
def zeroes(num_rows, num_cols=-1):
    """
    If only rows is provided, returns a matrix of all zeroes of size rows x rows. If cols is provided, returns same
    of size rows x cols.
    :param num_rows: integer (number of rows)
    :param num_cols: optional, integer (number of columns)
    :return: a matrix filled with zeroes
    """
    if num_cols == -1:
        countrows = range(num_rows)
        return [[0 * i * j for i in countrows] for j in countrows]
    else:
        countrows = range(num_rows)
        countcols = range(num_cols)
        return [[0 * i * j for i in countcols] for j in countrows]

## CONVERTED
def ones(num_rows, num_cols=-1):
    """
    As zeroes(), but a matrix filled with ones.
    :param num_rows: integer (number of rows)
    :param num_cols: optional, integer (number of columns)
    :return: a matrix filled with ones
    """
    if num_cols == -1:
        count_rows = range(num_rows)
        return [[1 ** i ** j for i in count_rows] for j in count_rows]
    else:
        count_rows = range(num_rows)
        count_cols = range(num_cols)
        return [[1 ** i ** j for i in count_cols] for j in count_rows]


def choose_one(state_vector):
    """
    Given a valid state vector, presumably in superposition and-or entangled, calculates the measurement
    probabilities and then returns a valid basis vector, randomly chosen based on those weights. Intended
    to simulate measurement of a quantum system.
    :param state_vector:
    :return:
    """
    norm_state = normalize(state_vector)  # Probabilities not valid if state not normalized
    probabilities = [abs(each) ** 2 for each in flatten(norm_state)[0]]
    options = list_possible_states(density_matrix(state_vector))
    selection = random.choices(options, weights=probabilities)[0]
    return selection

##CNVERTED
def eye(integer):
    """
    Gives a square eye matrix of size integer
    :param integer: a positive integer
    :return: a square eye matrix
    """
    return [[1 if i == j else 0 for i in range(integer)] for j in range(integer)]

## CONVERTED
def controlled(unitary):
    """
    Creates the matrix which does the operation of a controlled unitary gate
    :param unitary: a 2x2 unitary matrix
    :return: a 4x4 controlled unitary matrix
    """
    a = unitary[0][0]
    b = unitary[0][1]
    c = unitary[1][0]
    d = unitary[1][1]
    return [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, a, b],
        [0, 0, c, d],
    ]


def show(matrix):
    """
    Shows common matrices in pretty form, or shows row by row, or just prints for anything that isn't a matrix
    :param matrix: a matrix (or anything printable)
    :return: None
    """
    for key, value in Matrices.displays.items():
        if matrix == value:
            print(key)
            return None
    else:
        try:
            for each in matrix:
                print(each)
        except TypeError:
            print(matrix)
    return None

## CONVERTED TO OPERATOR
def matrix_matrix(matrices):
    """
    Computes the products of matrices
    :param matrices: iterable containing two or more matrices
    :return: a matrix
    """
    if len(matrices) == 2:
        matrix1 = matrices[0]
        matrix2 = matrices[1]
        m = range(rows(matrix1))
        k = range(columns(matrix2))
        matr = [[dot(get_row(matrix1, index), get_column(matrix2, jndex)) for jndex in k] for index in m]
        return matr
    else:
        return matrix_matrix([matrices[0], matrix_matrix(matrices[1:])])

## converted
def dot(vector1, vector2):
    """
    Computes the dot product of two vectors. Assumes vectors are valid sizes/orientation to be dotted.
    :param vector1: a vector
    :param vector2: a vector of same length but any orientation
    :return: a scalar
    """
    vec1 = conjugate_matrix(flatten(vector1))
    vec2 = flatten(vector2)
    count = range(columns(vec1))
    return sum([vec1[0][i] * vec2[0][i] for i in count])

##converted
def columns(matrix):
    """
    Gives the number of columns in a matrix
    :param matrix: a matrix
    :return: a positive integer
    """
    return len(matrix[0])

##converted
def rows(matrix):
    """
    Gives the number of rows in a matrix
    :param matrix: a matrix
    :return: a positive integer
    """
    return len(matrix)

## CIONVERTED TO OPERATOR
def scalar_matrix(scalar, matrix):
    """
    Computes the product of a matrix and a scalar
    :param scalar: a float or int of complex number
    :param matrix: a matrix
    :return: a matrix
    """
    count_cols = range(columns(matrix))
    count_rows = range(rows(matrix))
    return [[scalar * matrix[i][j] for j in count_cols] for i in count_rows]

##converted
def x(vector):
    """
    Does the X (NOT) operation on a vector
    :param vector: a column vector
    :return: a column vector
    """
    return matrix_matrix([Matrices.PauliX, vector])

##converted
def y(vector):
    """
    Does the Y operation on a vector
    :param vector: a column vector
    :return: a column vector
    """
    return matrix_matrix([Matrices.PauliY, vector])

##converted
def z(vector):
    """
    Does the Z operation on a vector
    :param vector: a column vector
    :return: a column vector
    """
    return matrix_matrix([Matrices.PauliZ, vector])

##converted
def hadamard(vector):
    """
    Does the Hadamard operation on a vector
    :param vector: a column vector
    :return: a column vector
    """
    return matrix_matrix([Matrices.Hadamard, vector])

##converted
def cnot(vector):
    """
    Does the CNOT operation on a vector
    :param vector: a column vector of length 4
    :return: a column vector of length 4
    """
    return matrix_matrix([Matrices.CNOT, vector])

##converted
def swap(state_vector, i):
    """
    Swaps two adjacent qubits and returns a state vector with appropriate size
    :param state_vector: any state vector
    :param i: the index of the first qubit in the swap, e.g. to swap qubits 2, 3, set i=2
    :return: a new state vector
    """
    length = len(state_vector)
    qubits = int(math.log(length, 2))
    ops = list()
    for _ in range(i):
        ops += [eye(2)]  # Do identity ops on all qubits prior to swap
    ops += [Matrices.SWAP]
    while len(ops) < qubits - 1:
        ops += [eye(2)]  # Do identity ops on all qubits after the swap
    return matrix_matrix([kronecker(ops) if len(ops) > 1 else ops[0], state_vector])

##converted
def flatten(vector):
    return [[element for row in vector for element in row]]

## CONVERTED
def transpose(matrix):
    """
    Computes the tranpose of any matrix
    :param matrix: a matrix
    :return: a transposed matrix
    """
    return [[matrix[j][i] for j in range(rows(matrix))] for i in range(columns(matrix))]


def collapse(vector, index):
    return vector, index

## CONVERTED
def add(matrices):
    """
    Computes the sum of two matrices
    :param matrices: an ordered iterable containing matrices
    :return: a matrix
    """
    if len(matrices) == 2:
        matrix1 = matrices[0]
        matrix2 = matrices[1]
        return [[matrix1[i][j] + matrix2[i][j] for j in range(columns(matrix1))] for i in range(rows(matrix2))]
    else:
        return add([matrices[0], add(matrices[1:])])

## CONVRTED
def subtract(matrix1, matrix2):
    """
    Computes the difference of two matrices
    :param matrix1: a matrix
    :param matrix2: a matrix
    :return: a matrix
    """
    return [[matrix1[i][j] - matrix2[i][j] for j in range(columns(matrix1))] for i in range(rows(matrix2))]

## converted
def norm(vector):
    """
    Returns the norm of a given vector (row or column)
    :param vector: any vector
    :return: a number
    """
    return math.sqrt(sum([abs(element) ** 2 for element in flatten(vector)[0]]))


def normalize(vector):
    """
    Returns the normalized vector
    :param vector: any vector
    :return: a normalized vector
    """
    the_norm = norm(vector)
    if the_norm != 0:
        return scalar_matrix(1 / norm(vector), vector)
    else:
        print("Cannot normalize zero vector")
        return vector


def list_possible_states(matrix):
    """
    Returns all the possible states on measuring a density matrix in the computational basis
    :param matrix: the nxn density matrix of a quantum state
    :return: a list of valid states (i.e. columns of length 2^n for some n)
    """
    r = range(rows(matrix))
    return [transpose([[1 if index == each else 0 for each in r]]) for index in r]


def measure(state_vector):
    """
    Takes a state (i.e. column) vector and returns the probability of measuring each of the valid computational basis
    states, by first calculatng the density matrix and then performing a projective values measurement.
    :param state_vector: a valid state vector, i.e. a column with length 2^n for some n
    :return: a list of strings
    """
    possible_measurements = list_possible_states(density_matrix(state_vector))
    probabilities = list()
    for each in possible_measurements:
        prob = pvm(state_vector, each)
        probabilities.append(f"Probability of measuring state |{str(each.index([1]))}> is {round(prob.real*100,1)}%.")
    return probabilities


def kronecker(matrices):
    """
    Recursively calculates the Kronecker product of two or more matrices.

    For two matrices A, B:

    A = [A11, A12],  B = [B11, B12]
        [A21, A22]       [B21, B22]

        A kron B = [A11.B, A12.B] = [A11.B11, A11.B12, A12.B11, A12.B12]
                   [A21.B, A22.B]   [A11.B21, A11.B22, A12.B21, A12.B22]
                                    [A21.B11, A21.B12, A22.B11, A22.B12]
                                    [A21.B21, A21.B22, A22.B21, A22.B22]

    :param matrices: An ordered iterable containing matrices.
    :return: A matrix.
    """
    if len(matrices) == 2:
        matrix1 = matrices[0]
        matrix2 = matrices[1]
        count = range(len(matrix2))
        return [[num1 * num2 for num1 in elem1 for num2 in matrix2[row]] for elem1 in matrix1 for row in count]
    else:
        return kronecker([matrices[0], kronecker(matrices[1:])])


def concat_horiz(matrix1, matrix2):
    """
    Concatenates two matrices horizontally, assuming the sizes allow it
    :param matrix1: a matrix
    :param matrix2: a matrix
    :return: a matrix
    """
    return [matrix1[row] + matrix2[row] for row in range(rows(matrix1))]


def concat_vert(matrix1, matrix2):
    """
    Concatenates two matrices vertically, assuming the sizes allow it
    :param matrix1: a matrix
    :param matrix2: a matrix
    :return: a matrix
    """
    return [matrix1[row] for row in range(rows(matrix1))] + [matrix2[row] for row in range(rows(matrix2))]


## Converted
def trace(matrix):
    """
    Computes the trace (sum along diagonal) of a square matrix
    :param matrix: a matrix
    :return: a float or int or complex number
    """
    return sum([matrix[i][i] for i in range(rows(matrix))])


def inverse(matrix):
    """
    Computes the inverse of a square matrix
    :param matrix: a square matrix
    :return: a square matrix
    """
    sign_matrix = cofactor_sign_matrix(matrix)
    minors_matrix = minors(matrix)
    composed_matrix = hadamard_product(sign_matrix, minors_matrix)
    transposed_matrix = transpose(composed_matrix)
    deter = determinant(matrix)
    if deter == 0:
        return "Non-invertible"
    final_matrix = scalar_matrix(1 / deter, transposed_matrix)
    return final_matrix


def minors(matrix):
    """
    Computes the matrix of minors of a given matrix
    :param matrix: a matrix
    :return: a matrix of minors
    """
    new_matrix = zeroes(rows(matrix), columns(matrix))
    for index in range(rows(matrix)):
        for jndex in range(rows(matrix)):
            new_matrix[index][jndex] = determinant(get_submatrix(
                matrix,
                [row for row in range(rows(matrix)) if row != index],
                [col for col in range(columns(matrix)) if col != jndex]
            ))
    return new_matrix


def hadamard_product(matrix1, matrix2):
    """
    Computes the element-wise product of two matrices
    :param matrix1: a matrix
    :param matrix2: a matrix
    :return: a matrix
    """
    return [[matrix1[i][j] * matrix2[i][j] for j in range(columns(matrix1))] for i in range(rows(matrix2))]


def hadamard_quotient(matrix1, matrix2):
    """
    Computes the element-wise quotient of two matrices. Can something useful be done in the ZeroDivisionError case?
    :param matrix1: a matrix
    :param matrix2: a matrix
    :return: a matrix
    """
    try:
        return [[matrix1[i][j] / matrix2[i][j] for j in range(columns(matrix1))] for i in range(rows(matrix2))]
    except ZeroDivisionError:
        return None


def cofactor_sign_matrix(matrix):
    """
    Gives a matrix of 1s and -1s of size equal to matrix
    :param matrix: a matrix
    :return: a matrix
    """
    return [[(-1) ** (i + j) for i in range(rows(matrix))] for j in range(rows(matrix))]


def is_number(thing):
    """
    Returns True if thing is a number, and false otherwise
    :param thing: any object
    :return: bool
    """
    return any([isinstance(thing, int), isinstance(thing, float), isinstance(thing, complex)])


def determinant(matrix):
    """
    Computes the determinant of a square matrix
    :param matrix: a square matrix
    :return: a number
    """
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 1
    new_matrix = [row for row in matrix]
    for i in range(1, rows(matrix)):
        if new_matrix[0][0] == 0:
            new_matrix = row_swap(new_matrix, 0, i)
            det *= -1
    if new_matrix[0][0] == 0:
        return 0
    condensed = condense(new_matrix)
    return det * condensed[0][0] * determinant(get_submatrix(condensed, range(1, len(matrix)), range(1, len(matrix))))


def power(matrix, apower):
    """
    Returns a matrix raised to a positive integer power.
    :param matrix:
    :param apower:
    :return:
    """
    if apower == 0:
        return eye(rows(matrix))
    if apower == 1:
        return matrix
    if apower == 2:
        return matrix_matrix([matrix, matrix])
    else:
        return matrix_matrix([matrix, power(matrix, apower - 1)])


def is_unitary(matrix):
    """
    Returns True if a given matrix is unitary, and False otherwise
    :param matrix: a square matrix
    :return: bool
    """
    return is_normal(matrix) and matrix_matrix([matrix, adjoint(matrix)]) == eye(len(matrix))


def is_normal(matrix):
    """
    Returns True if given matrix is normal, and false otherwise
    :param matrix: a square matrix
    :return: bool
    """
    return matrix_matrix([matrix, adjoint(matrix)]) == matrix_matrix([adjoint(matrix), matrix])


def conjugate_matrix(matrix):
    """
    Computes the complex conjugate of a matrix
    :param matrix: matrix
    :return: matrix
    """
    return [[element.conjugate() if isinstance(element, complex) else element for element in row] for row in matrix]


def adjoint(matrix):
    """
    Returns the complex conjugate tranpose (also called adjoint) of a matrix
    :param matrix: matrix
    :return: matrix
    """
    return conjugate_matrix(transpose(matrix))


def get_row(matrix, row_number):
    """
    Returns the requested row of the given matrix (zero-indexed)
    :param matrix: matrix
    :param row_number: positive integer
    :return: row vector
    """
    return [matrix[row_number]]


def get_column(matrix, column_number):
    """
    Returns the requested column of the given matrix (zero-indexed)
    :param matrix: matrix
    :param column_number: positive integer
    :return: column vector
    """
    count = range(rows(matrix))
    return [[matrix[i][column_number]] for i in count]


def get_rows(matrix, row_numbers):
    """
    Same as get_row, but returns multiple rows
    :param matrix: matrix
    :param row_numbers: iterable containing positive integers
    :return: matrix
    """
    return [matrix[i] for i in row_numbers]


def get_columns(matrix, column_numbers):
    """
    Same as get_column but returns multiple columns
    :param matrix: matrix
    :param column_numbers: iterable containing positive integers
    :return: matrix
    """
    count = range(rows(matrix))
    return [[matrix[i][column_number] for column_number in column_numbers] for i in count]


def get_submatrix(matrix, row_nums, col_nums):
    """
    Returns a submatrix of given matrix
    :param matrix: matrix
    :param row_nums: iterable of desired row numbers
    :param col_nums: iterable of desired column numbers
    :return: matrix
    """
    return get_columns(get_rows(matrix, row_nums), col_nums)


def condense(matrix):
    """
    Condenses a matrix, i.e. uses row operations to set elements [0][i] to zero for all i except i = 0
    :param matrix: matrix
    :return: condensed matrix
    """
    new_matrix = [matrix[0]] + [
        [matrix[i][j] - matrix[i][0] / matrix[0][0] * matrix[0][j] for j in range(columns(matrix))] for i in
        range(1, rows(matrix))]
    return new_matrix


def row_swap(matrix, row1, row2):
    """
    Swaps two rows in a matrix; a common elementary operation
    :param matrix: a matrix
    :param row1: a row number
    :param row2: a row number
    :return: a matrix
    """
    count = range(rows(matrix))
    return [matrix[i] if i not in [row1, row2] else matrix[row1] if i == row2 else matrix[row2] for i in count]


def random_matrix(num_rows, num_cols=-1):
    """
    Generates a random matrix of the given size
    :param num_rows: number of rows
    :param num_cols: number of columns, optional
    :return: a matrix
    """
    if num_cols == -1:
        return [[round(random.random(), 1) for row in range(num_rows)] for row in range(num_rows)]
    else:
        return [[round(random.random(), 1) for col in range(num_cols)] for row in range(num_rows)]


def triangular(matrix):
    """
    TODO: Not finished. Need to work out how to embed a new submatrix in the original.
    :param matrix:
    :return:
    """
    new_matrix = condense(matrix)
    sub = condense(get_submatrix(new_matrix, [1, 2], [1, 2]))
    return [new_matrix, sub]


def eigenvalues(matrix):
    #TODO
    return matrix


def eigenvectors(matrix):
    #TODO
    return matrix


def density_matrix(state_vector):
    """
    Generates a density matrix to represent a quantum state vector, appropriately normalized.
    :param state_vector:
    :return:
    """
    return kronecker([normalize(adjoint(state_vector)), normalize(state_vector)])


def pvm(state_vector, measurement):
    """
    Gives the probability of measuring state_vector to be in the state measurement
    :param state_vector:
    :param measurement:
    :return:
    """
    state = density_matrix(state_vector)
    return matrix_matrix([adjoint(measurement), state, measurement])[0][0]

## converted
def round(number, n):
    if not isinstance(number, complex):
        return python_round(number, n)
    else:
        return round(number.real, n) + round(number.imag, n) * Matrices.i

## converted
def qft(n, precision=10):
    """
    Creates the normalized quantum fourier transform matrix of size n, with entries rounded to precision
    :param n: positive integer
    :param precision: optional, rounding for the entries
    :return: qft matrix of size n
    """
    omega = math.cos(2 * math.pi / n) + Matrices.i * math.sin(2 * math.pi / n)
    matrix = [[round(omega ** (i * j) / math.sqrt(n), precision) for j in range(n)] for i in range(n)]
    return matrix

##converted
def mean_inversion_matrix(size):
    """
    Creates a matrix to compute a unitary inversion about the mean.
    :param size: the size of the inversion matrix
    :return: the inversion matrix
    """
    base_matrix = ones(size)
    identity = eye(size)
    two_over_n_matrix = scalar_matrix(2 / size, base_matrix)
    return subtract(two_over_n_matrix, identity)


def test1():
    count00 = 0
    count01 = 0
    count10 = 0
    count11 = 0
    for _ in range(100000):
        measurement = choose_one(Matrices.plus_plus)
        if measurement == [[1], [0], [0], [0]]:
            count00 += 1
        elif measurement == [[0], [1], [0], [0]]:
            count01 += 1
        elif measurement == [[0], [0], [1], [0]]:
            count10 += 1
        elif measurement == [[0], [0], [0], [1]]:
            count11 += 1
    print(
        f"|00>: {count00/1000}%,",
        f"|01>: {count01/1000}%,",
        f"|10>: {count10/1000}%,",
        f"|11>: {count11/1000}%.",
        sep="\n"
    )


def pull_substate():
    #TODO
    pass


def put_substate():
    #TODO
    pass


def swap_0_2(state_vector):
    """
    0 ----\ /-----------\ /---- 2
           X             X
    1 ----/ \----\ /----/ \---- 1
                  X
    2 -----------/ \----------- 0
    :param state_vector:
    :return:
    """
    return matrix_matrix([
        kronecker([Matrices.SWAP, eye(2)]),
        kronecker([eye(2), Matrices.SWAP]),
        kronecker([Matrices.SWAP, eye(2)]),
        state_vector
    ])

