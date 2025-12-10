from math import sqrt, pi
import math
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class Assortment(dict):
    def __init__(self, input_dict=None):
        super().__init__({k: v for k, v in input_dict if v > 0})

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        if len(self) != len(other):
            return False
        if self.keys() != other.keys():
            return False

        for k in self.keys():
            if self[k] != other[k]:
                return False
        return True


def add_equality_flag(circ, reg, C_bits, flag_qubit):
    """
    Toggle flag_qubit (assumed |0>) iff reg == C_bits.
    This is its own inverse: calling it again uncomputes.
    """
    # Encode pattern so that C corresponds to all-1s
    for i, bit in enumerate(C_bits):
        if bit == "0":
            circ.x(reg[i])

    circ.mcx(reg, flag_qubit)  # relative-phase is OK here because we uncompute

    # Undo encoding
    for i, bit in enumerate(C_bits):
        if bit == "0":
            circ.x(reg[i])


def add_phase_if_equal(circ, reg, C_bits: str):
    """Apply a phase -1 iff reg == C_bits, using no explicit ancilla."""
    # Encode pattern: flip qubits where C has '0' so that C -> '11...1'
    for i, bit in enumerate(C_bits):
        if bit == "0":
            circ.x(reg[i])

    # Multi-controlled Z on |11...1>
    if len(reg) == 1:
        circ.z(reg[0])
    else:
        last = reg[-1]
        ctrls = reg[:-1]
        circ.h(last)
        circ.mcx(ctrls, last)  # relative-phase MCX
        circ.h(last)

    # Undo pattern encoding
    for i, bit in enumerate(C_bits):
        if bit == "0":
            circ.x(reg[i])


def or_oracle(circ, qL, qR, C_bits: str):
    """
    Phase oracle for (L == C_bits) OR (R == C_bits).

    Implemented as product of three ancilla-free phase oracles:
      - O_L: phase flip if L == C
      - O_R: phase flip if R == C
      - O_LR: phase flip if L == C and R == C
    """
    # -1 if L == C
    add_phase_if_equal(circ, qL, C_bits)

    # -1 if R == C
    add_phase_if_equal(circ, qR, C_bits)

    # -1 if L == C and R == C (treat [L,R] as one big register equal to C+C)
    all_qubits = list(qL) + list(qR)
    C_concat = C_bits + C_bits
    add_phase_if_equal(circ, all_qubits, C_concat)


def diffusion(circ, qL, qR):
    """Standard Grover diffusion over the joint (L,R) register."""
    all_qubits = list(qL) + list(qR)

    # H then X on all
    for q in all_qubits:
        circ.h(q)
        circ.x(q)

    # Multi-controlled Z on |0...0> in this transformed basis
    last = all_qubits[-1]
    ctrls = all_qubits[:-1]
    circ.h(last)
    circ.mcx(ctrls, last)
    circ.h(last)

    # Undo X then H
    for q in all_qubits:
        circ.x(q)
        circ.h(q)


def build_multi_instance_or_circuit(
    C_list, num_iterations: int | None = None
) -> QuantumCircuit:
    """
    Build one big circuit that runs multiple independent OR-Grover
    gadgets in parallel, one for each C_bits in C_list.

    Condition per instance j:
        mark states where (L_j == C_j) OR (R_j == C_j).

    C_list: list of bitstrings, e.g. ["01", "11", ...].
            All must have same length Nc.

    Returns a circuit that measures each L_j and R_j pair into
    separate classical registers cL_j, cR_j.
    """
    if not C_list:
        raise ValueError("C_list must contain at least one bitstring.")

    # Ensure all patterns same length
    Nc = len(C_list[0])
    if any(len(C_bits) != Nc for C_bits in C_list):
        raise ValueError("All C bitstrings must have the same length.")

    num_instances = len(C_list)

    # Choose a common Grover iteration count if not provided
    if num_iterations is None:
        # Same N, M as single-instance OR rule
        N = 2 ** (2 * Nc)
        M = 2 * (2**Nc) - 1
        num_iterations = int((pi / 4) * sqrt(N / M))
        if num_iterations < 1:
            num_iterations = 1

    # Register containers
    L_regs = []
    R_regs = []
    cL_regs = []
    cR_regs = []

    qc = QuantumCircuit()

    # 1. Allocate quantum + classical registers per instance
    for j in range(num_instances):
        qL = QuantumRegister(Nc, f"L{j}")
        qR = QuantumRegister(Nc, f"R{j}")
        cL = ClassicalRegister(Nc, f"cL{j}")
        cR = ClassicalRegister(Nc, f"cR{j}")

        qc.add_register(qL)
        qc.add_register(qR)
        qc.add_register(cL)
        qc.add_register(cR)

        L_regs.append(qL)
        R_regs.append(qR)
        cL_regs.append(cL)
        cR_regs.append(cR)

    # 2. Superposition: for each instance, put (L_j, R_j) into uniform superposition
    for j in range(num_instances):
        qL = L_regs[j]
        qR = R_regs[j]
        for i in range(Nc):
            qc.h(qL[i])
            qc.h(qR[i])

    # 3. Grover iterations: oracle + diffusion for each instance
    for _ in range(num_iterations):
        # Oracles
        for j, C_bits in enumerate(C_list):
            qL = L_regs[j]
            qR = R_regs[j]
            or_oracle(qc, qL, qR, C_bits)

        # Diffusion
        for j in range(num_instances):
            qL = L_regs[j]
            qR = R_regs[j]
            diffusion(qc, qL, qR)

    # 4. Measurements
    for j in range(num_instances):
        qc.measure(L_regs[j], cL_regs[j])
        qc.measure(R_regs[j], cR_regs[j])

    return qc


def diffusion_on_subset(circ, qubits):
    """
    Grover diffusion operator on an arbitrary list of qubits.
    If the list is empty or length 1, handle degenerate cases.
    """
    if not qubits:
        return

    all_qubits = list(qubits)

    # H then X on all
    for q in all_qubits:
        circ.h(q)
        circ.x(q)

    if len(all_qubits) == 1:
        # Single-qubit diffusion: Z on |0> (in this transformed basis)
        last = all_qubits[0]
        circ.h(last)
        circ.z(last)
        circ.h(last)
    else:
        last = all_qubits[-1]
        ctrls = all_qubits[:-1]
        circ.h(last)
        circ.mcx(ctrls, last)
        circ.h(last)

    # Undo X then H
    for q in all_qubits:
        circ.x(q)
        circ.h(q)


def build_chunk(
    C_list_layer, extremas, num_iterations: int | None = None
) -> QuantumCircuit:
    """
    Build one Grover 'chunk' layer with OR constraints:

        For each j:
            (L_j == C_j) OR (R_j == C_j)

    with boundary conditions:
      - L_0 is pinned to extremas[0] and not measured.
      - R_{N-1} is pinned to extremas[1] and not measured.

    All instances still participate in the oracle, so the constraints
    L00 == C00 or R00 == C00
    L_last == C_last or R_last == C_last
    are enforced in the quantum circuit.

    Parameters
    ----------
    C_list_layer0 : list[str]
        List of bitstrings C_j for this layer, e.g. ["00","00","00","00"].
    extremas : list[str]
        [ext_left, ext_right] such that:
          - L_0 is prepared as ext_left.
          - R_{N-1} is prepared as ext_right.
    num_iterations : int or None
        Number of Grover iterations; if None, use near-optimal estimate
        based on the OR rule (for the interior full (L,R) pairs).
    """
    if not C_list_layer:
        raise ValueError("C_list_layer0 must contain at least one bitstring.")
    if extremas is None or len(extremas) != 2:
        raise ValueError("extremas must be a list [ext_left, ext_right].")

    N = len(C_list_layer)
    Nc = len(C_list_layer[0])

    if any(len(C_bits) != Nc for C_bits in C_list_layer):
        raise ValueError("All C bitstrings in the layer must have the same length.")
    if len(extremas[0]) != Nc or len(extremas[1]) != Nc:
        raise ValueError(
            "extremas must have the same bit-length as C_list_layer0 entries."
        )

    # Grover iteration count (approx, using the interior (L,R) pair space size)
    if num_iterations is None:
        N_states = 2 ** (2 * Nc)
        M_marked = 2 * (2**Nc) - 1  # OR rule
        num_iterations = int((pi / 4) * sqrt(N_states / M_marked))
        if num_iterations < 1:
            num_iterations = 1

    qc = QuantumCircuit()

    # Quantum registers per instance j
    L_regs = []
    R_regs = []

    # Classical registers: we will skip L_0 and R_{N-1}
    cL_regs = [None] * N
    cR_regs = [None] * N

    for j in range(N):
        qL = QuantumRegister(Nc, f"L{j}")
        qR = QuantumRegister(Nc, f"R{j}")
        qc.add_register(qL)
        qc.add_register(qR)
        L_regs.append(qL)
        R_regs.append(qR)

    # Classical registers:
    #  - measure R_j for j = 0..N-2 (skip R_{N-1})
    #  - measure L_j for j = 1..N-1 (skip L_0)
    for j in range(N):
        if j < N - 1:
            cR = ClassicalRegister(Nc, f"cR{j}")
            qc.add_register(cR)
            cR_regs[j] = cR
        if j > 0:
            cL = ClassicalRegister(Nc, f"cL{j}")
            qc.add_register(cL)
            cL_regs[j] = cL

    # 1. Initialize L_0 and R_{N-1} to their extremas; put others in superposition.
    ext_left, ext_right = extremas

    for j in range(N):
        qL = L_regs[j]
        qR = R_regs[j]

        if j == 0:
            # Pin L_0 to ext_left: encode bitstring via X gates.
            for i, bit in enumerate(ext_left):
                if bit == "1":
                    qc.x(qL[i])
        else:
            # Put L_j into uniform superposition (search variable)
            for i in range(Nc):
                qc.h(qL[i])

        if j == N - 1:
            # Pin R_{N-1} to ext_right
            for i, bit in enumerate(ext_right):
                if bit == "1":
                    qc.x(qR[i])
        else:
            # Put R_j into uniform superposition
            for i in range(Nc):
                qc.h(qR[i])

    # 2. Grover iterations
    for _ in range(num_iterations):
        # 2a. Apply OR oracle for each instance (including j=0 and j=N-1)
        for j, C_bits in enumerate(C_list_layer):
            qL = L_regs[j]
            qR = R_regs[j]
            or_oracle(qc, qL, qR, C_bits)

        # 2b. Apply diffusion only on "variable" qubits:
        #   - For j = 0: only R_0 is variable.
        #   - For j = N-1: only L_{N-1} is variable.
        #   - For 0 < j < N-1: both L_j and R_j are variables.
        for j in range(N):
            if j == 0:
                variable_qubits = list(R_regs[j])  # L_0 is pinned
            elif j == N - 1:
                variable_qubits = list(L_regs[j])  # R_{N-1} is pinned
            else:
                variable_qubits = list(L_regs[j]) + list(R_regs[j])

            diffusion_on_subset(qc, variable_qubits)

    # 3. Measure all variable L_j and R_j
    for j in range(N):
        if cR_regs[j] is not None:
            qc.measure(R_regs[j], cR_regs[j])
        if cL_regs[j] is not None:
            qc.measure(L_regs[j], cL_regs[j])

    return qc


def build_or_grover_circuit(
    C_bits: str, num_iterations: int | None = None
) -> QuantumCircuit:
    """
    Grover search over (L,R) such that (L == C_bits) OR (R == C_bits).
    """
    Nc = len(C_bits)

    # Choose near-optimal iteration count (you may want to recompute M for OR)
    if num_iterations is None:
        N = 2 ** (2 * Nc)
        # For OR, the number of marked (L,R) is:
        #  M = (#L==C)*all_R + (#R==C)*all_L - (#L==C and R==C)
        #    = (1 * 2**Nc) + (1 * 2**Nc) - 1 = 2*(2**Nc) - 1
        M = 2 * (2**Nc) - 1
        num_iterations = int((pi / 4) * sqrt(N / M))
        if num_iterations < 1:
            num_iterations = 1

    # Quantum registers
    qL = QuantumRegister(Nc, "L")
    qR = QuantumRegister(Nc, "R")

    # Classical registers
    cL = ClassicalRegister(Nc, "cL")
    cR = ClassicalRegister(Nc, "cR")

    qc = QuantumCircuit(qL, qR, cL, cR)

    # 1. Superposition on L and R
    for i in range(Nc):
        qc.h(qL[i])
        qc.h(qR[i])

    # 2. Grover iterations
    for _ in range(num_iterations):
        or_oracle(qc, qL, qR, C_bits)
        diffusion(qc, qL, qR)  # your existing diffusion over qL+qR

    # 3. Measure
    qc.measure(qL, cL)
    qc.measure(qR, cR)

    return qc


def decode_multi_counts(counts, Nc, num_instances, top_k=20):
    """
    Decode Qiskit bitstrings into logical (L_j, R_j) per instance.

    Assumes classical registers were added as:
        cL0, cR0, cL1, cR1, ..., cL_{n-1}, cR_{n-1}
    So Qiskit prints them as:
        cR_{n-1}, cL_{n-1}, ..., cR0, cL0
    and bits inside each register are reversed.
    """

    def fix_reg(s: str) -> str:
        # reverse bits inside each Nc-bit block
        return s[::-1]

    res = []
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:top_k]:
        parts = bitstring.split()  # e.g. ["00","01","10","00"]
        # Qiskit order: [cR_{n-1}, cL_{n-1}, ..., cR0, cL0]
        # Let's reverse register order to [cL0, cR0, cL1, cR1, ...]
        parts_rev = list(reversed(parts))

        pairs = []
        for j in range(num_instances):
            cL = parts_rev[2 * j]  # cL_j
            cR = parts_rev[2 * j + 1]  # cR_j
            L = fix_reg(cL)
            R = fix_reg(cR)
            pairs.append((L, R))

        res.append([pairs, count])
    return res


def build_multi_instance_circuit_with_nest(C_layers, num_iterations: int | None = None):
    """
    Build one multi-instance Grover circuit per layer.

    """
    circuits = []
    for layer_idx, C_list in enumerate(C_layers):
        qc = build_multi_instance_or_circuit(C_list, num_iterations=num_iterations)
        circuits.append(qc)

    return {"circuits": circuits, "C_layers": C_layers}


from qiskit_aer import AerSimulator
from qiskit import transpile


def pairs_decoder(k, extremas):
    if extremas is None:
        return tuple([rev[::-1] for rev in reversed(k.split(" "))])
    else:
        pairs = tuple(
            [extremas[0]]
            + [rev[::-1] for rev in reversed(k.split(" "))]
            + [extremas[1]]
        )
        return pairs


def process_dual_layer(
    C_list,
    C_list_next,
    extremas,
    C_list_prev: List | None = None,
    prev_set_extremas: set | None = None,
    assortment: Assortment | None = None,
):

    # now the next layer
    # extremas = ["11", "11"]
    # C_list = ["00", "11", "11"]
    # C_list_1 = ["00", "10"]
    # prev_layer_valid_pairs = valid_pairs if len(valid_pairs) > 0 else None
    if extremas is None:
        qc = build_multi_instance_or_circuit(C_list)
    else:
        qc = build_chunk(C_list, extremas=extremas)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=4096 * 8).result()
    counts = result.get_counts()
    valid_pairs1 = []
    max_count = 0

    extrema_set = dict()

    for k, v in sorted(counts.items(), key=lambda x: -x[1]):

        pairs = pairs_decoder(k, extremas)

        # if pairs is correct in assortment
        if assortment is not None:
            asso = get_pair_assortment(pairs)
            if asso != assortment:
                continue
        # if pairs is valid with the following layer
        if C_list_next is not None:
            valid = True
            for cidx, c_l1 in enumerate(C_list_next):
                if pairs[2 * cidx + 1] != c_l1 and pairs[2 * cidx + 2] != c_l1:
                    valid = False
                    break
            if not valid:
                continue

        # if pairs is valid with the previous layer
        if C_list_prev is not None:
            valid = True
            for cidx, c_l1 in enumerate(C_list_prev):
                if pairs[2 * cidx + 1] != c_l1 and pairs[2 * cidx + 2] != c_l1:
                    valid = False
                    break
            if not valid:
                continue

        # if extrema is None we want to check if the pairs extrema is within
        # the previous layer valid extremas
        if extremas is None and prev_set_extremas is not None:
            if (pairs[0], pairs[-1]) not in prev_set_extremas:
                continue

        if v > max_count:
            max_count = v
        valid_pairs1.append((pairs, (v / max_count)))

        extrema = (pairs[0], pairs[-1])
        if extrema not in extrema_set:
            extrema_set[extrema] = v
        else:
            extrema_set[extrema] += v

    if len(extrema_set) > 0:
        extrema_set = list(sorted(extrema_set.items(), key=lambda x: -x[1]))
        max_count_extrema = extrema_set[0][1] if len(extrema_set) > 0 else 1

        return valid_pairs1, [(k, v / max_count_extrema) for k, v in extrema_set]
    else:
        return valid_pairs1, []


from rich.text import Text
from rich.color import Color

COLOR_DICT = {
    "00": "blue",
    "01": "green",
    "10": "yellow",
    "11": "red",
}


def get_pair_assortment(pairs: List[str]) -> Assortment:
    from collections import Counter

    return Assortment(Counter(pairs).items())


def pairs_to_rich(pairs: List[str], value=None) -> Text:
    paris_str = " ".join([f"[{c}]|[/{c}]" for c in [COLOR_DICT[p] for p in pairs]])

    return Text.from_markup(f"{paris_str} > {value}")


def clist_to_rich(C_list: List[str]) -> Text:
    C_str = "   ".join([f"[{c}]#[/{c}]" for c in [COLOR_DICT[p] for p in C_list]])

    if len(C_list) % 2 == 1:
        C_str = "  " + C_str
    return Text.from_markup(f" {C_str}")
