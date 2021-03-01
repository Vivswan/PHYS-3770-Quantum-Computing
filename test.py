from QuantumComputer import QuantumComputer

r = {}

for i in range(0, 1000):

    qc = QuantumComputer(3)
    qc.H(0)
    qc.CNOT(0, 1)
    qc.H(1)
    qc.X(0)
    qc.CNOT(1, 2)
    qc.X(1)
    qc.Y(1)
    qc.Y(2)

    if i == 0:
        print(qc.bra_notation(probabilites=True))

    t = tuple(qc.measurement_all())
    if t not in r:
        r[t] = 0
    r[t] += 1

print(r)
