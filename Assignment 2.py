from QuantumComputer import QuantumComputer


def main():
    qc_a = QuantumComputer(2)
    qc_a.X(0)
    qc_a.CNOT(0)

    qc_b = QuantumComputer(3)
    qc_b.H(0)
    qc_b.CNOT(0)

    print("2 (A): ", end='')
    qc_a.print_wavefunction()

    print("2 (B): ", end='')
    qc_b.print_wavefunction()


if __name__ == '__main__':
    main()
