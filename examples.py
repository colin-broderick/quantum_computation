from operators import Matrix, Matrices, Vectors, Operators, Numbers
from qmath import exp
import math
import time

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


# def new_example1():
#     state = new_entangle_2q()
#     print("Chance of measuring |00>: ", end="")
#     print(state.pvm(Matrix([1], [0], [0], [0])))
#     print("Chance of measuring |01>: ", end="")
#     print(state.pvm(Matrix([0], [1], [0], [0])))
#     print("Chance of measuring |10>: ", end="")
#     print(state.pvm(Matrix([0], [0], [1], [0])))
#     print("Chance of measuring |11>: ", end="")
#     print(state.pvm(Matrix([0], [0], [0], [1])))


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
    ## TODO Flesh this example out some
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

    return register3.measure()


def rng():
    """
    Acting the Hadamard on vector |0> gives |0>+|1>, i.e.
    a 50/50 chance of measuring each basis state.
    """
    state = Operators.Hadamard * Vectors.zero
    result = state.measure()
    print(result)


def grover():
    register1 = Vectors.zero % Vectors.zero % Vectors.zero
    operator1 = Operators.Hadamard % Operators.Hadamard % Operators.Hadamard

    register2 = operator1 * register1
    
    pass


def shor():
    H = Operators.Hadamard
    I = Matrices.eye(2)
    X = Operators.PauliX

    print("Generating QFT")
    QFT = Matrices.qft(2**8)
    
    print("Inverting QFT")
    start = time.time()
    # QFTi = QFT.inverse
    print("QFT inverse took", time.time()-start, "seconds")

    print("Generting amodn operators")
    start = time.time()
    _7_1_mod15 = Operators.amodn(7, 1)
    _7_2_mod15 = Operators.amodn(7, 2)
    _7_4_mod15 = Operators.amodn(7, 4)
    _7_8_mod15 = Operators.amodn(7, 8)
    _7_16_mod15 = Operators.amodn(7, 16)
    _7_32_mod15 = Operators.amodn(7, 32)
    _7_64_mod15 = Operators.amodn(7, 64)
    _7_128_mod15 = Operators.amodn(7, 128)
    print("Generating amodn operators took", time.time()-start, "seconds")

    print("Generating initial register")
    start = time.time()
    register = Vectors.binary_vector(0, 12)
    print("Generating initial register took", time.time()-start, "seconds")
    
    print("Building and applying first operator")
    start = time.time()
    operator = H
    for _ in range(1, 8):
        operator = operator % H
    for _ in range(3):
        operator = operator % I
    operator = operator % X
    register = operator * register
    print("Building and applying first operator took", time.time()-start, "seconds")

    print("Generating and applying controlled unitaries")
    start = time.time()
    register = Matrices.controlled(0, 8, _7_1_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(1, 8, _7_2_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(2, 8, _7_4_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(3, 8, _7_8_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(4, 8, _7_16_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(5, 8, _7_32_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(6, 8, _7_64_mod15)*register
    print("--", time.time()-start)
    register = Matrices.controlled(7, 8, _7_128_mod15)*register
    print("--", time.time()-start)
    print("Generating and applying controlled unitaries took", time.time()-start, "seconds")

    operator = QFT % I % I % I % I
    register = operator * register

    results = register.measure()
    for result in results:
        print(result)


def shor_wrong():
    zero = Vectors.zero
    one = Vectors.one
    H = Operators.Hadamard
    I = Matrices.eye(2)
    X = Operators.PauliX
    p90 = Matrices.phase(math.pi/2)
    p45 = Matrices.phase(math.pi/4)

    ## Initialize as |0001111>
    register = zero % zero % zero % one % one % one % one

    ## Hadamards on first three bits, identity on others.
    operator = H % H % H % I % I % I % I
    register = operator*register

    ## CNOT(2, 4).
    operator = I % I % Matrices.controlled(0, 2, X) % I % I
    register = operator * register

    ## CNOT(2, 5).
    operator = I % I % Matrices.controlled(0, 3, X) % I
    register = operator*register

    ## CNOT(3, 5).
    operator = I % I % I % Matrices.controlled(0, 2, X) % I
    register = operator*register

    ## CNOT(1, 5, 3).
    operator = I % Matrices.controlled([0, 4], 2, X)  % I
    register = operator * register

    ## CNOT(3, 5).
    operator = I % I % I % Matrices.controlled(0,2,X) % I
    register = operator* register

    ## CNOT(6, 4).
    operator = I % I % I % I % Matrices.controlled(2, 0, X)
    register = operator* register

    ## CNOT(1, 4, 6).
    operator = I % Matrices.controlled([0,3], 5, X)
    register = operator*register

    ## CNOT(6, 4).
    operator = I % I % I % I % Matrices.controlled(2, 0, X)
    register = operator * register

    ## Hadamard on bit 0, identity otherwise
    operator = H % I % I % I % I % I % I
    register = operator * register

    ## Bit-0-controlled 90 degree phase gate on bit 1.
    operator = Matrices.controlled(0, 1, p90) % I % I % I % I % I
    register = operator * register

    ## Hadamard on bit 1.
    operator = I % H % I % I % I % I % I
    register = operator * register

    ## Bit-0-controlled 45 degree phase gate on bit 2.
    operator = Matrices.controlled(0, 2, p45) % I % I % I % I
    register = operator * register

    ## Bit-1-controlled 90 degree phase gate on bit 2.
    operator = I % Matrices.controlled(0, 1, p90) % I % I % I % I
    register = operator * register

    ## Hadamard on bit 2.
    operator = I % I % H % I % I % I % I
    register = operator * register

    ## Measure!
    results = register.measure()
    for result in results:
        print(result)


print(" Deutsch-Josza algorithm ".center(45, "-"))
results = new_deutsch("constant")
for result in results:
    print(result)
print()

# print(" 2-qubit entanglement ".center(45, "-"))
# results = new_entangle_2q()
# for result in results:
#     print(result)
# print()

# print(" Create |00> + |11> ".center(45, "-"))
# new_example1()
# print()

print(" Multi-qubit Deutsch-Josza algorithm ".center(45, "-"))
result = new_deutsch_multi("balanced")
for each in result:
    print(each)
print()

print(" Swapping adjacent qubits ".center(45, "-"))
swap_example()
print()

print(" Creating controlled gates ".center(45, "-"))
control()
print()

print(" Creating Bell ZZ state ".center(45, "-"))
results = bellzz()
for result in results:
    print(result)
print()


rng()


print("-"*45)

shor()

