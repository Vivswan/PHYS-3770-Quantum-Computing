# Source Code: https://github.com/Vivswan/PHYS-3770-Quantum-Information
#
# Output:
# 2 (A): Wavefunction = 1.000 |11>
# 2 (B): Wavefunction = 0.707 |00> + 0.707 |11>
#

from QuantumComputer import QuantumComputer


def main():
    qc_a = QuantumComputer(2)
    qc_a.X(0)
    qc_a.CNOT(0, 1)
    print("2 (A): ", qc_a.bra_notation())

    qc_b = QuantumComputer(2)
    qc_b.H(0)
    qc_b.CNOT(0, 1)
    print("2 (B): ", qc_b.bra_notation())


if __name__ == '__main__':
    main()
