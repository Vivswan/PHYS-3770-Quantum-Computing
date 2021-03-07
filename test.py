from QC.QuantumComputer import *

qc = QuantumComputer(2)
state = qc.state
qc.H(0)
qc.CNOT(0, 1)
qc.H(1)
qc.X(0)
qc.CNOT(1, 2)
qc.X(1)
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
uni = qc.unitary
initial = state
final = qc.state
qc.set_state(qc.state)
unitary = qc.unitary
print(np.round(uni, 2))
print(np.round(unitary, 2))
print(is_unitary(uni), np.isclose(np.matmul(uni, initial), final).all())
print(is_unitary(unitary), np.isclose(np.matmul(unitary, initial), final).all())
print(np.isclose(np.matmul(uni, initial), final).all(), np.isclose(uni, unitary).all())
