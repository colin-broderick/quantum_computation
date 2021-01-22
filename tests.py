import unittest
from operators import Matrix, Matrices, Vectors, Operators, Numbers


class MatrixMethods(unittest.TestCase):
    def test_conjugate(self):
        m1 = Matrices.eye(2)
        m2 = Matrices.eye(2)
        self.assertEqual(m1, m2)

        m1 = Operators.PauliY
        m2 = Operators.PauliY.conjugate
        self.assertEqual(m1, m2)

        m1 = Operators.SWAP * Numbers.i
        m2 = (Operators.SWAP * Numbers.i).conjugate
        self.assertEqual(m1, m2)

    def test_swap(self):
        m1 = Vectors.zero % Vectors.one
        m2 = Vectors.one % Vectors.zero
        m3 = m2.swap(0)
        self.assertEqual(m1, m3)

        m1 = Vectors.zero % Vectors.zero % Vectors.one
        m2 = Vectors.zero % Vectors.one % Vectors.zero
        m3 = m2.swap(1)
        self.assertEqual(m1, m3)

        m1 = Vectors.one % Vectors.zero % Vectors.zero
        m2 = Vectors.zero % Vectors.one % Vectors.zero
        m3 = m2.swap(0)
        self.assertEqual(m1, m3)

        m1 = Vectors.one % Vectors.zero % Vectors.zero % Vectors.plus % Vectors.minus
        m2 = Vectors.zero % Vectors.one % Vectors.zero % Vectors.plus % Vectors.minus
        m3 = m2.swap(0)
        self.assertEqual(m1, m3)

        m4 = Vectors.zero % Vectors.one % Vectors.zero % Vectors.minus % Vectors.plus
        m5 = m2.swap(3)
        self.assertEqual(m4, m5)



    def test_trace(self):
        trace = Operators.SWAP.trace
        self.assertEqual(2, trace)

        trace = Matrices.ones(7).trace
        self.assertEqual(7, trace)

        trace = Matrices.zeroes(19).trace
        self.assertEqual(0, trace)

        # matrix = Matrices.random(19)
        # trace1 = matrix.trace
        # trace2 = sum([matrix[i][i] for i in range(min(matrix.columns, matrix.rows)]))

        trace = Operators.SWAP.trace
        self.assertEqual(2, trace)

        trace = Operators.SWAP.trace
        self.assertEqual(2, trace)

if __name__ == "__main__":
    unittest.main()