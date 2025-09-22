# ce_systems.py
# J.S.C.
# Core implementation of Causal Emergence calculations

import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

def validate_tpm(matrix, tolerance=1e-4):
    assert isinstance(matrix, np.ndarray), "Input must be a numpy array"
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Matrix must be 2D and square"
    assert np.all(matrix >= 0), "Matrix not non-negative"
    assert np.allclose(matrix.sum(axis=1), 1, atol=tolerance), "Each row must sum to 1 within the given tolerance"
    return True

def is_permutation_state(tpm):
    n = tpm.shape[0]

    for row in tpm:
        if np.count_nonzero(row) == 1:
            continue
        else:
            return False
    
    return True

def mergeTPMDims(tpm, index1, index2):
    n = tpm.shape[0]
    i1, i2 = sorted((index1, index2))

    new_rows = []
    for i in range(n):
        if i == i1:
            new_rows.append((tpm[i1] + tpm[i2]) / 2.0) # average the two rows
        elif i == i2:
            continue
        else:
            new_rows.append(tpm[i])

    new_rows = np.vstack(new_rows)  # shape (n-1, n)

    final_tpm = []

    for row in new_rows:
        merged_row = []
        for j in range(n):
            if j == i1:
                # sum the two columns for the incoming to the merged state
                merged_row.append(row[i1] + row[i2])
            elif j == i2:
                continue
            else:
                merged_row.append(row[j])
        merged_row = np.array(merged_row, dtype=float)
        merged_row /= merged_row.sum() # just to make sure, should do essentially nothing but fixes fp
        final_tpm.append(merged_row)

    return np.vstack(final_tpm)

def merge_labels(labels, i, j):

    # Given a list of row‐labels (each a tuple of original indices),
    # merge labels[i] and labels[j] into one tuple, and drop the j‐entry.

    merged = tuple(sorted(labels[i] + labels[j]))
    new_labels = []
    for k, lab in enumerate(labels):
        if k == i:
            new_labels.append(merged)
        elif k == j:
            continue
        else:
            new_labels.append(lab)
    return new_labels

def calculate_cp_score(tpm, P_c=None):
    n = tpm.shape[0]

    # special case
    if n == 1:
        return 1.0, 1.0, 1.0

    if P_c is None:
        P_c = np.full(n, 1.0/n) # uniform prior if none given

    log2n = np.log2(n)  # strictly positive since n > 1
    H = lambda distribution: entropy(distribution, base=2)

    def _determinism():
        row_entropies = np.apply_along_axis(H, 1, tpm)
        weighted_average_row_entropies = P_c @ row_entropies
        return 1.0 - (weighted_average_row_entropies / log2n)

    def _specificity():
        weighted_column_sums = P_c @ tpm
        weighted_column_entropies = H(weighted_column_sums)
        degeneracy = 1.0 - (weighted_column_entropies / log2n)
        return 1.0 - degeneracy

    # compute metrics
    determinism = _determinism()
    specificity = _specificity()
    
    cp = (determinism + specificity) / 2.0

    return cp, determinism, specificity

# Find the best (i,j) pair to merge by maximizing CP.
def greedyMerge(tpm, labels, scorer=calculate_cp_score):
    n = tpm.shape[0]
    assert n > 1

    best_cp = -1.0
    cp_components = (-1.0, -1.0)
    best_tpm = None
    best_pair_idx = None
    best_pair_labels = None

    for i in range(n):
        for j in range(i + 1, n):
            candidate_tpm = mergeTPMDims(tpm, i, j)
            candidate_cp, candidate_det, candidate_spec = scorer(candidate_tpm)

            if candidate_cp > best_cp:
                best_cp = candidate_cp
                cp_components = (candidate_det, candidate_spec)
                best_tpm = candidate_tpm
                best_pair_idx = (i, j)
                # capture the two clusters (label‐tuples) being merged
                best_pair_labels = (labels[i], labels[j])

    validate_tpm(best_tpm)
    new_labels = merge_labels(labels, *best_pair_idx)
    return best_tpm, best_cp, cp_components, new_labels, best_pair_labels


# We want to feed a TPM and get a "path" from micro to macro...
# Want to keep some information. Most importantly the CP score of each state, but also which dimentions were merged.
# NOTICE: we can easily modify this to get a minimum...

def createPath(tpm, labels: List[Tuple[int, ...]] = None, scorer=calculate_cp_score):
    validate_tpm(tpm)

    if labels is None:
        labels = [(i,) for i in range(tpm.shape[0])]

    # initial microstate; no merge yet, so merged_labels = None
    micro_cp, micro_det, micro_spec = scorer(tpm)
    path = [(tpm, micro_cp, (micro_det, micro_spec), labels.copy(), None)]

    current_tpm = tpm
    current_labels = labels

    while current_tpm.shape[0] > 1:
        current_tpm, cp, (det, spec), current_labels, merged_labels = \
            greedyMerge(current_tpm, current_labels, scorer)
        path.append((current_tpm, cp, (det, spec), current_labels.copy(), merged_labels))

    return path

# TODO: In the end, I want this to be more modular, maybe an addStep function that takes in a TPM and returns the next step in the path.

def plot_cp_change(path):
    cp_values = [step[1] for step in path[:-1]]
    delta_cp_values = [0] + [cp_values[i] - cp_values[i-1] for i in range(1, len(cp_values))] + [0]

    plt.figure(figsize=(8, 4))
    steps = range(1, len(delta_cp_values) + 1)
    bars = plt.bar(steps, delta_cp_values, color='#999999')

    # Simpler scaling: max value is 3/4 of y-axis, with 25% padding
    max_val = max(abs(v) for v in delta_cp_values)
    if max_val == 0:
        y_max = 1
    else:
        y_max = max_val * 4/3  # max_val / 0.75 = max_val * 4/3

    plt.ylim(-y_max, y_max)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Dimension", fontsize=10)
    plt.ylabel("$\Delta$ CP", fontsize=10)
    plt.title("Change in CP across causal apportioning path", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.grid(False)
    plt.xticks(ticks=steps, labels=[i for i in reversed(steps)], fontsize=7)

    # Add value label above the tallest bar
    tallest_idx = np.argmax([abs(v) for v in delta_cp_values])
    tallest_bar = bars[tallest_idx]
    tallest_val = delta_cp_values[tallest_idx]
    plt.text(
        tallest_bar.get_x() + tallest_bar.get_width() / 2,
        tallest_bar.get_height() + (0.03 * y_max) * np.sign(tallest_val),
        f"{tallest_val:.2f}",
        ha='center', va='bottom' if tallest_val >= 0 else 'top',
        color='black', fontsize=9, fontweight='bold'
    )

    return plt

def plot_absolute_scores(path):
    cp_values = [step[1] for step in path]  # list the cp
    determinism_values = [step[2][0] for step in path]
    specificity_values = [step[2][1] for step in path]

    plt.figure(figsize=(8, 4))
    steps = range(1, len(cp_values) + 1)
    bars = plt.bar(steps, cp_values, color='#999999')
    
    plt.ylim(0, 1.25)  # Fixed y-axis max at 1
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Dimension", fontsize=10)
    plt.ylabel("Absolute CP", fontsize=10)
    plt.title("CP across causal apportioning path", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.grid(False)
    plt.xticks(ticks=steps, labels=[i for i in reversed(steps)], fontsize=7)
    
    # Add special labels for first bar and first bar that reaches 1.0
    # Label the first bar
    first_bar = bars[0]
    first_val = cp_values[0]
    plt.text(
        first_bar.get_x() + first_bar.get_width() / 2,
        first_bar.get_height() + 0.03,
        f"{first_val:.2f}",
        ha='center', va='bottom',
        color='black', fontsize=9, fontweight='bold'
    )
    
    # Label the first bar that reaches 1.0 (if any)
    first_one_idx = None
    for i, cp in enumerate(cp_values):
        if cp >= 1.0:
            first_one_idx = i
            break
    
    if first_one_idx is not None:
        first_one_bar = bars[first_one_idx]
        first_one_val = cp_values[first_one_idx]
        plt.text(
            first_one_bar.get_x() + first_one_bar.get_width() / 2,
            first_one_bar.get_height() + 0.03,
            f"{first_one_val:.2f}",
            ha='center', va='bottom',
            color='black', fontsize=9, fontweight='bold'
        )
    
    return plt

# given a sequence of paths, plot the absolute scores
# red is early in the sequence, blue is late
# these are line plot bars

def plot_sequence_absolute_scores(paths):

    plt.figure(figsize=(8, 4))
    
    n_paths = len(paths)
    
    for i, path in enumerate(paths):
        cp_values = [step[1] for step in path]
        steps = range(1, len(cp_values) + 1)
        
        # Color gradient from red to purple
        red_intensity = 1.0 - (i / max(1, n_paths - 1))
        blue_intensity = i / max(1, n_paths - 1)
        color = (red_intensity, 0, blue_intensity)
        
        plt.plot(steps, cp_values, color=color, linewidth=2, alpha=0.7, 
                marker='o', markersize=4, markerfacecolor=color, markeredgecolor=color)
    
    plt.ylim(0, 1.25)
    plt.axhline(0, color='black', linewidth=1)
    plt.axhline(1.0, color='lightgray', linewidth=1, linestyle='--')
    plt.xlabel("Dimension", fontsize=10)
    plt.ylabel("Absolute CP", fontsize=10)
    plt.title("CP across greedy sequence of causal apportioning paths", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.grid(False)
    
    # Reverse x-axis labels to match other plots
    max_steps = max(len(path) for path in paths)
    plt.xticks(ticks=range(1, max_steps + 1), 
               labels=[i for i in reversed(range(1, max_steps + 1))], 
               fontsize=7)
    
    return plt


def format_clusters(clusters: List[Tuple[int, ...]]) -> str:

    parts = []
    for cluster in clusters:
        # turn each element into a Python int then to string
        elems = ", ".join(str(int(x)) for x in cluster)
        parts.append(f"({elems})")
    return "[" + ", ".join(parts) + "]"

def plot_sequence_cp_change(paths):
    plt.figure(figsize=(8, 4))
    
    n_paths = len(paths)
    
    # calculate max value across all paths for consistent scaling
    all_deltas = []
    for path in paths:
        cp_values = [step[1] for step in path[:-1]]
        delta_cp_values = [0] + [cp_values[i] - cp_values[i-1] for i in range(1, len(cp_values))] + [0]
        all_deltas.extend(delta_cp_values)
    
    max_val = max(abs(v) for v in all_deltas) if all_deltas else 1
    y_max = max_val * 4/3 if max_val > 0 else 1
    
    for i, path in enumerate(paths):
        cp_values = [step[1] for step in path[:-1]]
        delta_cp_values = [0] + [cp_values[i] - cp_values[i-1] for i in range(1, len(cp_values))] + [0]
        steps = range(1, len(delta_cp_values) + 1)
        
        # Color gradient from red to purple
        red_intensity = 1.0 - (i / max(1, n_paths - 1))
        blue_intensity = i / max(1, n_paths - 1)
        color = (red_intensity, 0, blue_intensity)
        
        plt.plot(steps, delta_cp_values, color=color, linewidth=2, alpha=0.7, 
                marker='o', markersize=4, markerfacecolor=color, markeredgecolor=color)
    
    plt.ylim(-y_max, y_max)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("Dimension", fontsize=10)
    plt.ylabel("$\Delta$ CP", fontsize=10)
    plt.title("Change in CP across greedy sequence of causal apportioning paths", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.grid(False)
    
    # Reverse x-axis labels to match other plots
    max_steps = max(len(path) for path in paths)
    plt.xticks(ticks=range(1, max_steps + 1), 
               labels=[i for i in reversed(range(1, max_steps + 1))], 
               fontsize=7)
    
    return plt
