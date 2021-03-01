from QuantumComputer import QuantumComputer

qc = QuantumComputer(2)
qc.H(0)
qc.CNOT(0, 1)
qc.H(1)
# qc.X(0)
# # qc.CNOT(1, 2)
# qc.X(1)
# qc.Y(1)
# qc.Y(2)
# # qc.CNOT(2, 0)

# p = []
# for i in range(0, 3):
#     p.append(qc.get_probabilities_of(i))


print(qc.bra_notation())
print(qc.measurement(0))
print(qc.sim_measurement(0, get_states=True))
