#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_results():
    # vanilla
    recall_vanilla = [0.833, 0.920, 0.971, 0.992, 0.997, 0.999, 1.000, 1.000]
    blocks_vanilla = [1528.08, 2269.80, 3456.58, 5850.28, 8457.66, 14005.32, 26703.02, 49756.98]
    
    # k=3
    recall_k3 = [0.681, 0.817, 0.914, 0.968, 0.984, 0.994, 0.999, 1.000]
    blocks_k3 = [334.62, 538.66, 947.36, 1666.14, 2329.62, 3839.16, 7104.28, 13828.70]
    
    # k=5
    recall_k5 = [0.732, 0.854, 0.936, 0.980, 0.990, 0.997, 1.000, 1.000]
    blocks_k5 = [394.52, 643.32, 1063.78, 1806.14, 2653.72, 4179.56, 8030.42, 15997.70]
    
    # k=7
    recall_k7 = [0.764, 0.875, 0.948, 0.984, 0.993, 0.998, 1.000, 1.000]
    blocks_k7 = [462.50, 688.54, 1169.54, 2068.12, 2972.98, 4609.88, 9082.54, 17866.02]
    
    # Plot results
    plt.plot(recall_vanilla, blocks_vanilla, marker='s', color="black", linewidth=2, label="vanilla pgvector")
    plt.plot(recall_k7, blocks_k7, marker='o', color="tab:blue", label="w/patch(k=7)")
    plt.plot(recall_k5, blocks_k5, marker='^', color="tab:green", label="w/patch(k=5)")
    plt.plot(recall_k3, blocks_k3, marker='v', color="tab:orange", label="w/patch(k=3)")

    plt.xlabel("Recall")
    plt.ylabel("#Blocks")
    plt.yscale("log")
    plt.title("Recall-#Blocks tradeoff (SIFT1M,10-NN,m=24,ef_construction=200)")
    plt.legend()
    plt.grid(True, which="both")

    # Save to file
    plt.savefig("sift1m_recall_blocks_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_results()

