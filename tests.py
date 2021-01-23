import unittest
from operators import Matrix, Matrices, Vectors, Operators, Numbers


class MatrixMethods(unittest.TestCase):
    def test_conjugate(self):
        m1 = Matrices.eye(2)
        m2 = Matrices.eye(2)
        self.assertEqual(m1, m2)

        m1 = Operators.PauliY
        m2 = Operators.PauliY.conjugate
        m3 = Matrix([0, Numbers.i], [-Numbers.i, 0])
        self.assertEqual(m2, m3)

        m1 = Operators.SWAP * Numbers.i
        m2 = (Operators.SWAP * Numbers.i).conjugate
        self.assertEqual(m2, -m1)

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

    def test_multi_swap(self):
        m1 = Vectors.one % Vectors.zero % Vectors.zero % Vectors.plus % Vectors.minus
        m2 = Vectors.zero % Vectors.one % Vectors.zero % Vectors.plus % Vectors.minus
        m3 = m2.multi_swap(0, 1)
        self.assertEqual(m1, m3)

        m1 = Vectors.plus % Vectors.zero % Vectors.zero % Vectors.one % Vectors.minus
        m2 = Vectors.one % Vectors.zero % Vectors.zero % Vectors.plus % Vectors.minus
        m3 = m2.multi_swap(0, 3)
        self.assertEqual(m1, m3)

    def test_cofactor_signs(self):
        m1 = Matrix(
            [1, 3, 5, 9],
            [1, 3, 1, 7],
            [4, 3, 9, 7],
            [5, 2, 0, 9]
        )
        m2 = Matrix(
            [2, 4, 1],
            [7, 2, 2],
            [3, 3, 2],
        )
        m1_cof = Matrix(
            [ 1, -1,  1, -1],
            [-1,  1, -1,  1],
            [ 1, -1,  1, -1],
            [-1,  1, -1,  1]
        )
        m2_cof = Matrix(
            [1, -1, 1],
            [-1, 1, -1],
            [1, -1, 1]
        )
        self.assertEqual(m1.cofactor_signs, m1_cof)
        self.assertEqual(m2.cofactor_signs, m2_cof)

    def test_is_normal(self):
        self.assertTrue(Operators.PauliX.is_normal)
        self.assertTrue(Operators.PauliY.is_normal)
        self.assertTrue(Operators.PauliZ.is_normal)
        self.assertTrue(Operators.PauliI.is_normal)
        self.assertTrue(Operators.CNOT.is_normal)
        self.assertTrue(Operators.SWAP.is_normal)

    def test_is_unitary(self):
        self.assertTrue(Operators.PauliX.is_unitary)
        print()
        print(Operators.PauliY)
        print(Operators.PauliY.adjoint)
        print(Operators.PauliY.conjugate)
        print(Operators.PauliY.transpose)
        # self.assertTrue(Operators.PauliY.is_unitary)
        self.assertTrue(Operators.PauliZ.is_unitary)
        self.assertTrue(Operators.PauliI.is_unitary)
        self.assertTrue(Operators.CNOT.is_unitary)
        self.assertTrue(Operators.SWAP.is_unitary)

    def test_determinant(self):
        self.assertEqual(-1, Operators.PauliX.determinant)
        self.assertEqual(1, Matrices.eye(2).determinant)
        self.assertEqual(1, Matrices.eye(3).determinant)
        
        m1 = Matrix(
            [1, 3, 5, 9],
            [1, 3, 1, 7],
            [4, 3, 9, 7],
            [5, 2, 0, 9]
        )
        self.assertEqual(m1.determinant, -376)

        m2 = Matrix(
            [2, 4, 1, -3],
            [7, 2, 2, -2],
            [3, 3, 2, 2],
            [0, 5, 1, 0]
        )
        self.assertEqual(m2.determinant, -35)
        

    def test_CNOT(self):
        cnot = Operators.CNOT
        m2 = Vectors.zero_zero
        m3 = Vectors.zero_one
        m4 = Vectors.two
        m5 = Vectors.three
        self.assertEqual(m2, cnot*m2)
        self.assertEqual(m3, cnot*m3)
        self.assertEqual(m4, cnot*m5)
        self.assertEqual(m5, cnot*m4)

    def test_controlled(self):
        ## TODO: More tests are warranted here.
        m000 = Vectors.zero % Vectors.zero % Vectors.zero
        m001 = Vectors.zero % Vectors.zero % Vectors.one
        m010 = Vectors.zero % Vectors.one % Vectors.zero
        m011 = Vectors.zero % Vectors.one % Vectors.one
        m100 = Vectors.one % Vectors.zero % Vectors.zero
        m101 = Vectors.one % Vectors.zero % Vectors.one
        m110 = Vectors.one % Vectors.one % Vectors.zero
        m111 = Vectors.one % Vectors.one % Vectors.one
        
        op = Operators.PauliX

        self.assertEqual(Matrices.controlled(0, 2, op)*m000, m000)
        self.assertEqual(Matrices.controlled(0, 2, op)*m001, m001)
        self.assertEqual(Matrices.controlled(0, 2, op)*m010, m010)
        self.assertEqual(Matrices.controlled(0, 2, op)*m011, m011)
        self.assertEqual(Matrices.controlled(0, 2, op)*m100, m101)
        self.assertEqual(Matrices.controlled(0, 2, op)*m101, m100)
        self.assertEqual(Matrices.controlled(0, 2, op)*m110, m111)
        self.assertEqual(Matrices.controlled(0, 2, op)*m111, m110)
        
        
    def test_trace(self):
        # TODO: Incomplete
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
