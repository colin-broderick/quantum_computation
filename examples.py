from operators import Matrix, Matrices, Vectors, Operators


def new_entangle_2q():
    """
    This is a simple circuit for creating a maximal entanglement of two qubits.
    We prepare the state |00>, then apply a Hadamard gate to qubit 1, and a CNOT gate from qubit 1 to qubit 2.
    The result is state |00> + |11>, which is one of the four 2-qubit Bell states.
    :return: an entangled state.
    """    
    register0 = Vectors.zero % Vectors.zero

    operator1 = Operators.Hadamard % Matrices.eye(2)
    register1 = operator1 * register0

    register2 = Operators.CNOT * register1

    return register2




def new_deutsch(kind):
  
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



def new_example1():
    state = new_entangle_2q()
    print("Chance of measuring |00>: ", end="")
    print(state.pvm(Matrix([1], [0], [0], [0])))
    print("Chance of measuring |01>: ", end="")
    print(state.pvm(Matrix([0], [1], [0], [0])))
    print("Chance of measuring |10>: ", end="")
    print(state.pvm(Matrix([0], [0], [1], [0])))
    print("Chance of measuring |11>: ", end="")
    print(state.pvm(Matrix([0], [0], [0], [1])))


def new_deutsch_multi(kind):
    """
    TODO: This explanation isn't really sufficient unles you're already an expert.
    Explanation:
    If we use the constant operator, where f(x) = 1 for all x, then careful analysis shows that we will measure y = 0
    with certainty. Keep in mind y is the first three qubits only; we don't care about the state of the final qubit.
    If we use the balanced operator, then we must measure anything other than y = 0; at least one of the other
    possible measurements must therefore have a non-zero probability.    
    """

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



def swap_example():
    """
    We create a multi-qubit vector and then swap two chosen
    qubit states. In particular, we go from
      |0> |0> |+> |-> |0>
    to
      |0> |0> |-> |+> |0>
    """
    #TODO Rewrite this example now that it works
    # start_state = Vectors.zero % Vectors.zero % Vectors.plus % Vectors.minus % Vectors.zero
    # final_state = Vectors.zero % Vectors.zero % Vectors.minus % Vectors.plus % Vectors.zero
    start_state = Vectors.zero % Vectors.one % Vectors.zero % Vectors.zero % Vectors.zero % Vectors.zero
    final_state = Vectors.zero % Vectors.zero % Vectors.one % Vectors.zero % Vectors.zero % Vectors.zero

    test_state = start_state.swap(4)
    if ( test_state == final_state ):
        print("EQUAL")
    else:
        print("NEQUAL")



def control():
    register0 = Vectors.zero % Vectors.zero % Vectors.zero
    register1 = Vectors.one % Vectors.zero % Vectors.zero
    op = (Vectors.zero % Vectors.zero.transpose) % Matrices.eye(4) + (Vectors.one % Vectors.one.transpose) % Matrices.eye(2) % Operators.PauliX
    print((op*register0-register0).transpose)
    print()
    print((op*register1-register1).transpose)


def bellzz():
    register1 = Vectors.zero % Vectors.zero
    operator1 = Operators.Hadamard % Matrices.eye(2)
    register2 = operator1 * register1

    operator2 = Matrices.controlled(0, 1, Operators.PauliX)

    register3 = operator2 * register2
    print(register3.measure())


print(" Deutsch-Josza algorithm ".center(45, "-"))
(new_deutsch("constant"))
print()
print(" 2-qubit entanglement ".center(45, "-"))
(new_entangle_2q())
print()
print(" Create |00> + |11> ".center(45, "-"))
new_example1()
print()
print(" Multi-qubit Deutsch-Josza algorithm ".center(45, "-"))
(new_deutsch_multi("balanced"))
print()
print(" Swapping adjacent qubits ".center(45, "-"))
swap_example()
print("-"*45)


control()
bellzz()