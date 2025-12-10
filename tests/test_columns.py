from typing import List
import unittest
from qiskit_aer import AerSimulator
from qiskit import transpile

# from build0 import build_multi_instance_circuit


def add_hoc_reorder(bitstring: str) -> List[List[str]]:
    bits = bitstring.split(" ")  # will have 6 parts
    # group into pairs
    paired_bits = [[bits[i], bits[i + 1]] for i in range(0, len(bits), 2)]
    pair_correct_order = list(
        reversed(paired_bits)
    )  # reverse to get correct order (L0,R0), (L1,R1), (L2,R2)

    return pair_correct_order


class TestColumn4Color(unittest.TestCase):

    def run_test_instance(self, instance_count, idx):
        # <instance_count> classical inputs, all 2 bits
        C_list = ["00"] * instance_count
        C_list[idx] = "11"

        multi_qc = build_multi_instance_circuit(C_list)

        sim = AerSimulator()
        tqc = transpile(multi_qc, sim)
        result = sim.run(tqc, shots=2048).result()
        counts = result.get_counts()

        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
            LR_results = add_hoc_reorder(bitstring)

            self.assertEqual(len(LR_results), instance_count)
            self.assertIn("11", LR_results[idx])

    def test_4colors_3instances(self):
        instance_count = 3
        for idx in range(instance_count):
            with self.subTest(
                i=idx,
                instance_count=instance_count,
            ):
                self.run_test_instance(instance_count, idx)

    def test_man(self):
        instance_count = 4
        self.run_test_instance(instance_count, 2)
