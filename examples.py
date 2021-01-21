from operators import kronecker, Matrices, eye, measure, show, pvm, normalize, matrix_matrix


def new_entangle_2q():
    """
    This is a simple circuit for creating a maximal entanglement of two qubits.
    We prepare the state |00>, then apply a Hadamard gate to qubit 1, and a CNOT gate from qubit 1 to qubit 2.
    The result is state |00> + |11>, which is one of the four 2-qubit Bell states.
    :return: an entangled state.
    """
    from operators2 import Matrices, Operators, Vectors
    
    register0 = Vectors.zero % Vectors.zero

    operator1 = Operators.Hadamard % Matrices.eye(2)
    register1 = operator1 * register0

    register2 = Operators.CNOT * register1

    return register2


## converted
def entangle_2q():
    """
    This is a simple circuit for creating a maximal entanglement of two qubits.
    We prepare the state |00>, then apply a Hadamard gate to qubit 1, and a CNOT gate from qubit 1 to qubit 2.
    The result is state |00> + |11>, which is one of the four 2-qubit Bell states.
    :return: an entangled state.
    """
    # Initialize register 0
    register0 = [Matrices.zero, Matrices.zero]
    register0 = kronecker(register0)
    # Define the first operator
    operator1 = [Matrices.Hadamard, eye(2)]
    operator1 = kronecker(operator1)
    # Apply the first operator
    register1 = matrix_matrix([operator1, register0])
    # At this point the system is in a superposition state equal to |00>+|10> but it is not entangled.
    # define the second operator
    operator2 = Matrices.CNOT
    # Apply the second operator
    register2 = matrix_matrix([operator2, register1])
    # The system is now in a superposition state equal to |00>+|11> which is an entangled state; cannot be factored.
    return register2


def new_deutsch(kind):
    from operators2 import Matrix, Matrices, Vectors, Operators
    
    register1 = Vectors.zero % Vectors.minus
    
    operator1 = Operators.Hadamard % Matrices.eye(2)
    register2 = operator1 * register1
    
    if kind == "balanced":
        operator2 = Operators.DeutschBalanced
    else:
        operator2 = Operators.DeutschConstant
    register3 = operator2 * register2
    
    operator4 = operator1
    register4 = operator4 * register3
    
    result = register4.measure()
    
    return result


## converted
def deutsch_josza(type):
    types = {
        "constant": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],  # Here f(0) = f(1) = 1
        "balanced": [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]   # Here f(0) = 1 and f(1) = 0
    }
    # Initialize the state |01>
    register1 = [Matrices.zero, Matrices.minus]
    register1 = kronecker(register1)
    # Create and apply the H^2 operator to get state |+->
    operator1 = kronecker([Matrices.Hadamard, eye(2)])
    register2 = matrix_matrix([operator1, register1])
    # Apply the appropriate unitary operator
    operator2 = types[type]
    # Apply the unitary operator
    register3 = matrix_matrix([operator2, register2])
    # Apply the Hadamard to qubit 1 and leave qubit 2 alone
    register4 = matrix_matrix([kronecker([Matrices.Hadamard, eye(2)]), register3])
    # Perform measurements
    result = measure(register4)
    return result


def new_example1():
    from operators2 import Matrix

    state = new_entangle_2q()
    print("Chance of measuring |00>: ", end="")
    show(state.pvm(Matrix([1], [0], [0], [0])))
    print("Chance of measuring |01>: ", end="")
    show(state.pvm(Matrix([0], [1], [0], [0])))
    print("Chance of measuring |10>: ", end="")
    show(state.pvm(Matrix([0], [0], [1], [0])))
    print("Chance of measuring |11>: ", end="")
    show(state.pvm(Matrix([0], [0], [0], [1])))


## converted
def example1():
    """
    Demonstrates the creation of the state |00> + |11>
    :return:
    """
    state = entangle_2q()
    print("Chance of measuring |00>: ", end="")
    show(pvm(state, [[1], [0], [0], [0]]))
    print("Chance of measuring |01>: ", end="")
    show(pvm(state, [[0], [1], [0], [0]]))
    print("Chance of measuring |10>: ", end="")
    show(pvm(state, [[0], [0], [1], [0]]))
    print("Chance of measuring |11>: ", end="")
    show(pvm(state, [[0], [0], [0], [1]]))


def new_deutsch_multi(kind):
    """
    TODO: This explanation isn't really sufficient unles you're already an expert.
    Explanation:
    If we use the constant operator, where f(x) = 1 for all x, then careful analysis shows that we will measure y = 0
    with certainty. Keep in mind y is the first three qubits only; we don't care about the state of the final qubit.
    If we use the balanced operator, then we must measure anything other than y = 0; at least one of the other
    possible measurements must therefore have a non-zero probability.    
    """

    from operators2 import Matrix, Matrices, Operators, Vectors

    kinds = {
        "constant": Matrices.eye(2) % Matrices.eye(2) % Matrices.eye(2) % Operators.PauliX,
        "balanced": Matrices.eye(2) % Matrices.eye(2) % Operators.CNOT
    }

    register0 = Vectors.zero % Vectors.zero % Vectors.zero % Vectors.minus.normal

    operator0 = Operators.Hadamard % Operators.Hadamard % Operators.Hadamard % Matrices.eye(2)
    register1 = operator0 * register0

    operator1 = kinds[kind]
    register2 = operator1 * register1

    operator2 = operator0
    register3 = operator2 * register2

    result = register3.measure()

    return result


## converted
def deutsch_multi(type):
    types = {
        "constant": kronecker([eye(2), eye(2), eye(2), Matrices.PauliX]),  # Here f(x) = 1 for all x
        "balanced": kronecker([eye(2), eye(2), Matrices.CNOT])  # Here f(0) = 1 and f(1) = 0
    }
    # Register 0 is the initialization state of the circuit. Nothing much to see.
    register0 = kronecker([Matrices.zero, Matrices.zero, Matrices.zero, normalize(Matrices.minus)])

    # Operator 0 is the first operation we apply; Hadamard to the first three qubits, and identity to the fourth.
    operator0 = kronecker([Matrices.Hadamard, Matrices.Hadamard, Matrices.Hadamard, eye(2)])
    register1 = matrix_matrix([operator0, register0])

    # Choose and apply an operator based in user input.
    operator1 = types[type]
    register2 = matrix_matrix([operator1, register1])

    # This operator is just the 3xHadamard + identity again.
    operator2 = operator0
    register_final = matrix_matrix([operator2, register2])
    result = measure(register_final)
    """
    Explanation:
    If we use the constant operator, where f(x) = 1 for all x, then careful analysis shows that we will measure y = 0
    with certainty. Keep in mind y is the first three qubits only; we don't care about the state of the final qubit.
    If we use the balanced operator, then we must measure anything other than y = 0; at least one of the other
    possible measurements must therefore have a non-zero probability.    
    """
    return result


print(" Deutsch-Josza algorithm ".center(45, "-"))
print("Old")
show(new_deutsch("balanced"))
print("New")
show(new_deutsch("constant"))
print()
print(" 2-qubit entanglement ".center(45, "-"))
print("Old")
show(entangle_2q())
print("New")
show(new_entangle_2q())
print()
print(" Create |00> + |11> ".center(45, "-"))
print("Old")
example1()
print("New")
new_example1()
print()
print(" Multi-qubit Deutsch-Josza algorithm ".center(45, "-"))
print("Old:")
show(deutsch_multi("balanced"))
print("New:")
show(new_deutsch_multi("balanced"))
print()
print("-"*45)
