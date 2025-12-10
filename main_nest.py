from qiskit_aer import AerSimulator
from qiskit import transpile



def run_two_layer_nested(C_layers, backend=None, shots=2048):
    """
    Specialised for the 2-layer case:
    C_layers = [
        C_layer0,  # e.g. ["01","10","11"]
        C_layer1   # e.g. ["01","10"]  with len = len(C_layer0) - 1
    ]

    Enforces wiring:
      from layer0 (3 instances) to layer1 (2 instances):
        - R00 & L01 must equal C10
        - R01 & L02 must equal C11
    and returns a filtered joint distribution over (layer0, layer1) outputs.
    """
    if len(C_layers) != 2:
        raise ValueError("run_two_layer_nested currently assumes exactly 2 layers.")

    C0, C1 = C_layers
    n0 = len(C0)
    n1 = len(C1)
    if n1 != n0 - 1:
        raise ValueError(
            "Second layer must have len = first layer len - 1 for this wiring pattern."
        )

    Nc = len(C0[0])
    if any(len(c) != Nc for c in C0 + C1):
        raise ValueError("All C bitstrings in all layers must have the same length.")

    # Build circuits
    qc0 = build_multi_instance_circuit(C0)
    qc1 = build_multi_instance_circuit(C1)

    # Choose backend
    if backend is None:
        backend = AerSimulator()

    # Run both layers
    tqc0 = transpile(qc0, backend)
    tqc1 = transpile(qc1, backend)
    result0 = backend.run(tqc0, shots=shots).result()
    result1 = backend.run(tqc1, shots=shots).result()
    counts0 = result0.get_counts()
    counts1 = result1.get_counts()

    # Convert to probabilities
    total0 = sum(counts0.values())
    total1 = sum(counts1.values())
    probs0 = {k: v / total0 for k, v in counts0.items()}
    probs1 = {k: v / total1 for k, v in counts1.items()}
    # sort probabilities descending
    probs0 = sorted(probs0.items(), key=lambda x: -x[1])
    probs1 = sorted(probs1.items(), key=lambda x: -x[1])

    # Decode all outcomes
    decoded0 = {k: decode_layer_output(k, C0, Nc) for k, _ in probs0}
    decoded1 = {k: decode_layer_output(k, C1, Nc) for k, _ in probs1}

    # Apply cross-layer constraints
    filtered = {}
    for key0, p0 in probs0:
        layer0_out = decoded0[key0]

        # Extract relevant L/R from layer 0
        # Instances: 0,1,2
        L00 = layer0_out[0]["L"]
        R00 = layer0_out[0]["R"]
        L01 = layer0_out[1]["L"]
        R01 = layer0_out[1]["R"]
        L02 = layer0_out[2]["L"]
        R02 = layer0_out[2]["R"]

        # Constraints:
        # C10 = C1[0], C11 = C1[1]
        C10 = C1[0]
        C11 = C1[1]

        # Check wiring constraints from specification:
        #   R00 and L01 must equal C10
        #   R01 and L02 must equal C11
        if not (R00 == C10 and L01 == C10 and R01 == C11 and L02 == C11):
            continue  # discard this layer0 outcome entirely

        # Now combine with any layer1 outcome (they are statistically independent)
        for key1, p1 in probs1:
            joint_key = (key0, key1)
            joint_p = p0 * p1
            filtered[joint_key] = filtered.get(joint_key, 0.0) + joint_p

    return {
        "joint_distribution": filtered,
        "C_layers": C_layers,
        "counts_layer0": counts0,
        "counts_layer1": counts1,
    }


#
# C_layers = [
#    ["1", "1", "1"],  # layer 0 (odd)
#    ["0", "0"],  # layer 1 (even)
# ]
#
# nested = run_two_layer_nested(C_layers)
# print("Filtered joint outcomes (respecting wiring constraints):")
# for (key0, key1), p in sorted(
#    nested["joint_distribution"].items(), key=lambda x: -x[1]
# ):
#    print(f"Layer0: {key0}, Layer1: {key1}, P ≈ {p:.4f}")
#
