# ce_testing.py
# J.S.C.
# Some example TPMs for testing CE path apportionment.

import numpy as np
import networkx as nx

example1 = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0]
], dtype=float)

perm_8x8 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],  # state 0 → 1
    [0, 0, 1, 0, 0, 0, 0, 0],  # state 1 → 2
    [0, 0, 0, 1, 0, 0, 0, 0],  # state 2 → 3
    [0, 0, 0, 0, 1, 0, 0, 0],  # state 3 → 4
    [0, 0, 0, 0, 0, 1, 0, 0],  # state 4 → 5
    [0, 0, 0, 0, 0, 0, 1, 0],  # state 5 → 6
    [0, 0, 0, 0, 0, 0, 0, 1],  # state 6 → 7
    [1, 0, 0, 0, 0, 0, 0, 0]  # state 7 → 0
], dtype=float)

unif_4x4 = np.array([
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25]
], dtype=float)

unif_8x8 = np.array([
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 0 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 1 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 2 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 3 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 4 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 0 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],  # state 1 → all
    [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]  # state 2 → all
], dtype=float)

loop_5x5 = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],  # state 0 → 0
    [0.0, 1.0, 0.0, 0.0, 0.0],  # state 1 → 1
    [0.0, 0.0, 1.0, 0.0, 0.0],  # state 2 → 2
    [0.0, 0.0, 0.0, 1.0, 0.0],  # state 3 → 3
    [0.0, 0.0, 0.0, 0.0, 1.0]   # state 4 → 4
], dtype=float)

# high det, low spec
all_rivers_lead_to_the_sea = np.array([
    [0.0, 0.0, 0.0, 1.0],  # state 0 → 3
    [0.0, 0.0, 0.0, 1.0],  # state 1 → 0
    [0.0, 0.0, 0.0, 1.0],  # state 2 → 1
    [0.0, 0.0, 0.0, 1.0]   # state 3 → 2
], dtype=float)


# generate a random TPM for a Markov chain with n=8 states
def randTPM(n=8):    
    random_tpm_8x8 = np.random.rand(n, n)
    random_tpm_8x8 = random_tpm_8x8 / random_tpm_8x8.sum(axis=1, keepdims=True)
    return random_tpm_8x8

rand_8x8 = randTPM(8)
rand_20x20 = randTPM(20)
rand_40x40 = randTPM(40)
rand_100x100 = randTPM(100)

alternating_8x8 = np.empty((8, 8), dtype=float)
for i in range(8):
    if i % 2 == 0:
        alternating_8x8[i] = np.tile([0.0, 0.25], 4)
    else:
        alternating_8x8[i] = np.tile([0.25, 0.0], 4)

# Yes! this behaves how I thought it would!
block_12x12 = np.zeros((12, 12), dtype=float)
for b in range(3):
    i = b * 4
    block_12x12[i:i+4, i:i+4] = np.full((4, 4), 0.25)


# E-R Markov Chain!

def generate_markovian_er(n, p, seed) -> nx.DiGraph:
    rand = np.random.default_rng(seed)

    G = nx.fast_gnp_random_graph(n, p, seed=seed, directed=False)

    # pick oriented edges to randomly
    M = nx.DiGraph()
    M.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if rand.random() < 0.5:
            M.add_edge(u, v)
        else:
            M.add_edge(v, u)

    # normalize probs to 1
    for u in M.nodes():
        succ = list(M.successors(u))
        # I should make this non-uniform, but for now keep simple
        if succ:
            prob = 1.0 / len(succ)
            for v in succ:
                M[u][v]["weight"] = prob
        if succ:
            for v in succ:
                M[u][v]["weight"] = rand.integers(1, 101)
            # normalize weights
            total_weight = sum(M[u][v]["weight"] for v in succ)
            for v in succ:
                M[u][v]["weight"] /= total_weight
        else:
            M.add_edge(u, u, weight=1.0)

    return M

def chain_to_TPM(G: nx.DiGraph) -> np.ndarray:

    # gpt suggested:
    nodes = sorted(G.nodes())
    index_of = {node: idx for idx, node in enumerate(nodes)}

    tpm = np.zeros((len(nodes), len(nodes)), dtype=float)


    for u, v, data in G.edges(data=True):
        i = index_of[u]
        j = index_of[v]
        tpm[i, j] = data.get("weight", 0.0)

    return tpm

G = generate_markovian_er(n=12, p=0.1, seed=69)
random_markov_tpm = chain_to_TPM(G)


# Eriks example:
Erik_block_model = np.array([
    [0.21, 0.21, 0.21, 0.21, 0.04, 0.04, 0.04, 0.04],
    [0.21, 0.21, 0.21, 0.21, 0.04, 0.04, 0.04, 0.04],
    [0.21, 0.21, 0.21, 0.21, 0.04, 0.04, 0.04, 0.04],
    [0.21, 0.21, 0.21, 0.21, 0.04, 0.04, 0.04, 0.04],
    [0.04, 0.04, 0.04, 0.04, 0.21, 0.21, 0.21, 0.21],
    [0.04, 0.04, 0.04, 0.04, 0.21, 0.21, 0.21, 0.21],
    [0.04, 0.04, 0.04, 0.04, 0.21, 0.21, 0.21, 0.21],
    [0.04, 0.04, 0.04, 0.04, 0.21, 0.21, 0.21, 0.21],
], dtype=float)


Erik_mesoscale = np.array([
    [0.21, 0.00, 0.00, 0.00, 0.21, 0.21, 0.21, 0.21],
    [0.00, 0.07, 0.07, 0.07, 0.21, 0.21, 0.21, 0.21],
    [0.00, 0.07, 0.07, 0.07, 0.21, 0.21, 0.21, 0.21],
    [0.00, 0.07, 0.07, 0.07, 0.21, 0.21, 0.21, 0.21],
    [0.21, 0.21, 0.21, 0.21, 0.21, 0.00, 0.00, 0.00],
    [0.21, 0.21, 0.21, 0.21, 0.00, 0.07, 0.07, 0.07],
    [0.21, 0.21, 0.21, 0.21, 0.00, 0.07, 0.07, 0.07],
    [0.21, 0.21, 0.21, 0.21, 0.00, 0.07, 0.07, 0.07],
])
# this doesn't work, he made a mistake in his notes...


Erik_mesoscale_FIXED = np.array([
    [0.16, 0.00, 0.00, 0.00, 0.21, 0.21, 0.21, 0.21],
    [0.01, 0.05, 0.05, 0.05, 0.21, 0.21, 0.21, 0.21],
    [0.01, 0.05, 0.05, 0.05, 0.21, 0.21, 0.21, 0.21],
    [0.01, 0.05, 0.05, 0.05, 0.21, 0.21, 0.21, 0.21],
    [0.21, 0.21, 0.21, 0.21, 0.16, 0.00, 0.00, 0.00],
    [0.21, 0.21, 0.21, 0.21, 0.01, 0.05, 0.05, 0.05],
    [0.21, 0.21, 0.21, 0.21, 0.01, 0.05, 0.05, 0.05],
    [0.21, 0.21, 0.21, 0.21, 0.01, 0.05, 0.05, 0.05],
])

exemplar = np.array([
    [0.8, 0.2, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.1, 0.0, 0.9, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0]
], dtype=float)

Jesse_mesoscale = np.array([
    [0.2, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.2, 0.2],
    [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
    [0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.1, 0.0],
    [0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.1, 0.0],
    [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.2],
], dtype=float)

all_road_to_rome_mesoscale = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 0.4, 0.6]
])

# GPT suggested SBM Markov chain generator
def generate_sbm_markov_tpm(n_blocks, block_size, intra_alpha=0.8, p_intra=0.8, p_inter=0.05, seed_base=100):
    """
    Generates a Stochastic Block Model Markov chain TPM.
    - n_blocks: Number of blocks.
    - block_size: Number of nodes per block.
    - intra_alpha: Fractional weight multiplier for intra-block transitions.
    - p_intra: Probability parameter for generate_markovian_er() within blocks.
    - p_inter: Chance (per possible inter-block edge) to add an extra inter-block connection.
    - seed_base: Base seed for reproducibility.
    
    Returns:
        A TPM (numpy.ndarray) for the resulting Markov chain.
    """
    total_nodes = n_blocks * block_size
    global_graph = nx.DiGraph()
    global_graph.add_nodes_from(range(total_nodes))
    block_membership = {}

    # uild intra-block subgraphs and add them to the global graph.
    for b in range(n_blocks):
        nodes_block = list(range(b * block_size, (b + 1) * block_size))
        for node in nodes_block:
            block_membership[node] = b
        subG = generate_markovian_er(n=block_size, p=p_intra, seed=seed_base + b)
        # Add each intra-block edge with scaled weight.
        for u, v, data in subG.edges(data=True):
            global_u = nodes_block[u]
            global_v = nodes_block[v]
            global_graph.add_edge(global_u, global_v, weight=data["weight"] * intra_alpha)

    # Add inter-block edges.
    rng = np.random.default_rng(seed_base)
    for u in range(total_nodes):
        for v in range(total_nodes):
            if block_membership[u] != block_membership[v]:
                if rng.random() < p_inter:
                    w = rng.integers(1, 101)
                    if global_graph.has_edge(u, v):
                        global_graph[u][v]["weight"] += w
                    else:
                        global_graph.add_edge(u, v, weight=w)
        # Ensure each node has at least one outgoing edge; if not, add a self-loop.
        if global_graph.out_degree(u) == 0:
            global_graph.add_edge(u, u, weight=1.0)
        else:
            successors = list(global_graph.successors(u))
            total_weight = sum(global_graph[u][v]["weight"] for v in successors)
            for v in successors:
                global_graph[u][v]["weight"] /= total_weight

    # Convert the resulting graph to a TPM.
    return chain_to_TPM(global_graph)

example_sbm_tpm = generate_sbm_markov_tpm(n_blocks=5, block_size=3, intra_alpha=1.0, p_intra=0.8, p_inter=0.10, seed_base=42)

# EXTREMAL SYSTEMS

# HIGH DET, HIGH SPEC

# Permutation
perm_6x6 = np.array([
    [0, 1, 0, 0, 0, 0],  # state 0 → 1
    [0, 0, 1, 0, 0, 0],  # state 1 → 2
    [0, 0, 0, 1, 0, 0],  # state 2 → 3
    [0, 0, 0, 0, 1, 0],  # state 3 → 4
    [0, 0, 0, 0, 0, 1],  # state 4 → 5
    [1, 0, 0, 0, 0, 0]   # state 5 → 0
], dtype=float)
perm_graph = nx.from_numpy_array(perm_6x6, create_using=nx.DiGraph)

# Loop
loop_1x1 = np.array([
    [1]  # state 0 → 0
], dtype=float)
loop_graph = nx.from_numpy_array(loop_1x1, create_using=nx.DiGraph)

## LOW DET, HIGH SPEC

# Uniform rows, uniform columns
uniform_6x6 = np.array([
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] 
], dtype=float)
uniform_graph = nx.from_numpy_array(uniform_6x6, create_using=nx.DiGraph)

# HIGH DET, LOW SPEC

# All roads lead to rome
all_to_rome_6x6 = np.array([
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1]
], dtype=float)
all_to_rome_graph = nx.from_numpy_array(all_to_rome_6x6, create_using=nx.DiGraph)

# LOW DET, LOW SPEC
# Prove minimum, reverse-engineer from others

current_min_6x6 = np.array([
    [0.3682985,  0.13530801, 0.2328376,  0.17434466, 0.05459674, 0.03461449],
    [0.38484998, 0.15035649, 0.21755572, 0.15177204, 0.05729195, 0.03817381],
    [0.35370045, 0.16425028, 0.25463870, 0.14455105, 0.05110392, 0.03175560],
    [0.35555271, 0.14655419, 0.23273804, 0.18659936, 0.03794513, 0.04061059],
    [0.39269106, 0.14891351, 0.23056285, 0.14200754, 0.04243769, 0.04338734],
    [0.37639415, 0.15119797, 0.20766276, 0.17737309, 0.04186139, 0.04551064]
], dtype=float)
current_min_6x6_graph = nx.from_numpy_array(current_min_6x6, create_using=nx.DiGraph)


# BREAKING THE GREEDY ALGORITHM

try1 = np.array([
    [0.01, 0.01, 0.49, 0.49], 
    [0.05, 0.05, 0.45, 0.45],
    [0.49, 0.49, 0.01, 0.01],
    [0.78, 0.10, 0.06, 0.06]
], dtype=float)

try2 = np.array([
    [0.05, 0.05, 0.05, 0.05, 0.8,  0.0,  0.0,  0.0 ],  # 0 (A) strong -> 4
    [0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0 ],  # 1 (A)
    [0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0 ],  # 2 (A)
    [0.25, 0.25, 0.25, 0.25, 0.0,  0.0,  0.0,  0.0 ],  # 3 (A)
    [0.8,  0.0,  0.0,  0.0,  0.05, 0.05, 0.05, 0.05],  # 4 (B) strong -> 0
    [0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25],  # 5 (B)
    [0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25],  # 6 (B)
    [0.0,  0.0,  0.0,  0.0,  0.25, 0.25, 0.25, 0.25]  # 7 (B)
], dtype=float)

try3 = np.array([
    [0.3333333, 0.3333333, 0.3333333, 0.0,  0.0,  0.0],   # 0 (A)
    [0.05,      0.10,      0.05,      0.0,  0.8,  0.0],   # 1 (A) strong -> 4
    [0.3333333, 0.3333333, 0.3333333, 0.0,  0.0,  0.0],   # 2 (A)
    [0.0,       0.0,       0.0,       0.3333333, 0.3333333, 0.3333333],  # 3 (B)
    [0.0,       0.0,       0.0,       0.3333333, 0.3333333, 0.3333333],  # 4 (B)
    [0.0,       0.0,       0.8,       0.05,      0.05,      0.10],       # 5 (B) strong -> 2
], dtype=float)

try5 = np.array([
    [0.10, 0.05, 0.05, 0.40, 0.40, 0.00],  # 0 (A) -> mostly B (3,4)
    [0.05, 0.20, 0.05, 0.00, 0.70, 0.00],  # 1 (A) -> mostly 4
    [0.05, 0.05, 0.10, 0.80, 0.00, 0.00],  # 2 (A) **decoy** -> 3
    [0.00, 0.00, 0.80, 0.10, 0.05, 0.05],  # 3 (B) **decoy mate** -> 2
    [0.70, 0.00, 0.00, 0.05, 0.20, 0.05],  # 4 (B) -> mostly 0 (A)
    [0.40, 0.40, 0.00, 0.05, 0.05, 0.10],  # 5 (B) -> mostly A (0,1)
], dtype=float)

bait1 = np.array([
    [0.99, 0.01, 0.0, 0.0],
    [0.05, 0.0, 0.95, 0.0],
    [0.0, 0.95, 0.0, 0.05],
    [0.0, 0.0, 0.01, 0.99]
], dtype=float)

bait2 = np.array([
    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.4, 0.4, 0.0, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.2, 0.0, 0.4, 0.4],
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]
], dtype=float)


# INTUITION: WHEN DO WE MERGE FOR DET AND WHEN FOR SPEC?

merge_for_det = np.array([
    [0.50, 0.50, 0.00, 0.00],
    [0.00, 0.00, 0.50, 0.50],
    [0.25, 0.25, 0.25, 0.25],
    [0.25, 0.25, 0.25, 0.25]
], dtype=float)

merge_for_spec = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1]
], dtype=float)
