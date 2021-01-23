def collapse(vector, index):
    ## TODO
    return vector, index


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
