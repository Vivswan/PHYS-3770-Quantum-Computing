from QC.QuantumComputer import *

qc = QuantumComputer(2)
qc.H(0)
# qc.CNOT(0, 1)
# qc.H(1)
# qc.X(0)
qc.CNOT(1, 2)
# qc.X(1)
qc.Y(1)
qc.Y(2)

# qc.save()

# print(qc.ket_notation(probabilities=True))
#
# r = {}
#
# for i in range(0, 1000):
#     qc.load()
#
#     t = tuple(qc.measurement_all())
#     if t not in r:
#         r[t] = 0
#     r[t] += 1
#
# print(r)
# initial_state = zero_state_matrix(qc.num_qubit())
# initial_density_matrix = zero_state_matrix(qc.num_qubit(), 2)
#
# unitary_density_matrix = create_unitary(initial_density_matrix, qc.density_matrix)
# print("unitary:\n", np.round(qc.unitary, 2))
# print("unitary_density_matrix:\n", np.round(unitary_density_matrix, 2))
# print("dm == s: ", np.isclose(unitary_density_matrix, unitary_state).all())
# print("unitary: ", is_unitary(unitary_state), is_unitary(unitary_density_matrix))
# print("final_s: ",
#       np.isclose(matmul(unitary_state, initial_state), qc.state).all(),
#       np.isclose(matmul(qc.unitary, initial_state), qc.state).all(),
#       np.isclose(matmul(initial_state.conj().T, qc.unitary.conj().T).conj().T, qc.state).all(),
# )
# print("final_dm: ",
#       np.isclose(matmul(unitary_density_matrix, initial_density_matrix, unitary_density_matrix.conj().T), qc.density_matrix).all(),
#       np.isclose(matmul(unitary_state, initial_density_matrix, unitary_state.conj().T), qc.density_matrix).all(),
#       np.isclose(matmul(qc.unitary.conj().T, initial_density_matrix, qc.unitary), qc.density_matrix).all()
# )
# print(np.round(matmul(unitary_state.conj().T, initial_density_matrix, unitary_state), 2))
dm = density_matrix_to_state(qc.density_matrix)
print(np.round(qc.state, 2).tolist())
print(np.round(dm, 2))
print(np.isclose(dm, qc.state).all())
