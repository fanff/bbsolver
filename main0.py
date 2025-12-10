from qiskit_aer import AerSimulator
from qiskit import transpile

from qiskit.primitives import StatevectorSampler
from rich.text import Text
from build0 import (
    Assortment,
    build_chunk,
    build_multi_instance_or_circuit,
    build_or_grover_circuit,
    clist_to_rich,
    decode_multi_counts,
    pairs_to_rich,
    process_dual_layer,
)
from pprint import pprint
from rich.console import Console

console = Console()


assortment = Assortment({"00": 4, "01": 0, "10": 1, "11": 3}.items())
extremas = ["11", "11"]  # fixed L00 and R03 values.
C_list_layer0 = ["00", "00", "00", "00"]  # C00 , C01, C02, C03
# this is layer 0 with 4 instances yeilding 4 (L,R) outputs; minus the two existing fixed
# (L00(internal only),R00), (L01,R01), (L02,R02), (L03,R03(internal only))

# contraints in layer 0:
# L00 == C00 or R00 == C00
# L01 == C01 or R01 == C01
# L02 == C02 or R02 == C02
# L03 == C03 or R03 == C03


C_list_layer1 = ["11", "11", "11"]  # C10, C11, C12
# this is layer 1 with 3 instances yeilding 3 (L,R) outputs (L10,R10), (L11,R11), (L12,R12)

# contraints in layer 1:
# L10 == C10 or R10 == C10
# L11 == C11 or R11 == C11
# L12 == C12 or R12 == C12

# constraints between layer 0 and layer 1:
# R00 == C10 or L01 == C10
# R01 == C11 or L02 == C11
# R02 == C12 or L03 == C12

# L00 & R03 are fixed so we do NOT measure it, we know for sure their values.
sols, extrema_set = process_dual_layer(
    C_list_layer0, C_list_layer1, extremas=None, C_list_prev=None, assortment=assortment
)
console.print(clist_to_rich(C_list_layer0))
console.print(clist_to_rich(C_list_layer1))
for p, v in sols:
    console.print(pairs_to_rich(p, v))
print("\n\nExtrema sets:\n\n")
pprint(extrema_set)

# now the next layer
console.print("\n\nNext layer:\n\n")
# extremas = extrema_set[0][0]  # pick one extrema to fix L10 and R12
C_list_layer2 = ["00", "11", "10" ,"00"]
C_list_layer3 = ["00", "10", "11"]
sols2, extrema_set = process_dual_layer(
    C_list_layer2,
    C_list_layer3,
    extremas=None,
    C_list_prev=C_list_layer1,
    prev_set_extremas=set(e for e, _ in extrema_set),
    assortment=assortment,
)

console.print(clist_to_rich(C_list_layer2))
console.print(clist_to_rich(C_list_layer3))
for p, v in sols2:
    console.print(pairs_to_rich(p, v))

# now the next layer
console.print("\n\nNext 3 :\n\n")
C_list_layer4 = ["00", "11", "10", "00"]
C_list_layer5 = ["00", "10", "11"]
sols3, extrema_set = process_dual_layer(
    C_list_layer4,
    C_list_layer5,
    extremas=None,
    C_list_prev=C_list_layer3,
    prev_set_extremas=set(e for e, _ in extrema_set),
    assortment=assortment,
)
for p, v in sols3:
    console.print(pairs_to_rich(p, v))


print("\n\nFinal idea :\n\n")
console.print(pairs_to_rich(*sols[0]) + Text(f" len={len(sols)}"))
console.print(clist_to_rich(C_list_layer0))
console.print(clist_to_rich(C_list_layer1))
console.print(pairs_to_rich(*sols2[0]) + Text(f" len={len(sols2)}"))
console.print(clist_to_rich(C_list_layer2))
console.print(clist_to_rich(C_list_layer3))
console.print(pairs_to_rich(*sols3[0]) + Text(f" len={len(sols3)}"))
console.print(clist_to_rich(C_list_layer4))
console.print(clist_to_rich(C_list_layer5))
quit(0)


filtered_event_solutions = []
for pairs, count in decoded_even:
    valid = True
    for cidx, c_l1 in enumerate(C_list_layer1):
        if pairs[cidx][1] != c_l1 and pairs[cidx + 1][0] != c_l1:
            valid = False
            break
    if valid:
        filtered_event_solutions.append((pairs, count))
from pprint import pprint

pprint(filtered_event_solutions)

decoded_odd = process_layer(C_list_layer1)
quit(0)


C_bits = "1"  # your classical input C
qc = build_or_grover_circuit(C_bits, num_iterations=1)
print(qc)

sim = AerSimulator()
tqc = transpile(qc, sim)
result = sim.run(tqc, shots=2048).result()
counts = result.get_counts()

print("Measurement results (cL cR):")
for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(bitstring, "->", count)


exit(0)
