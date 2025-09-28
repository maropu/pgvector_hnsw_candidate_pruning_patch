#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_results():
    # vanilla
    recall_vanilla = [0.833, 0.920, 0.971, 0.992, 0.997, 0.999, 1.000, 1.000]
    blocks_vanilla = [1528.08, 2269.80, 3456.58, 5850.28, 8457.66, 14005.32, 26703.02, 49756.98]

    # k=1
    recall_k1 = [0.587, 0.763, 0.890, 0.960, 0.980, 0.993, 0.999, 1.000]
    blocks_k1 = [193.420, 271.100, 417.340, 698.560, 974.180, 1530.480, 2908.180, 5605.100]
    
    # k=3
    recall_k3 = [0.740, 0.870, 0.949, 0.985, 0.994, 0.998, 1.000, 1.000]
    blocks_k3 = [301.840, 424.980, 646.920, 1034.120, 1410.340, 2190.800, 4163.740, 8156.520]
    
    # k=5
    recall_k5 = [0.788, 0.900, 0.962, 0.990, 0.996, 0.999, 1.000, 1.000]
    blocks_k5 = [403.160, 570.720, 849.320, 1385.580, 1929.480, 3004.180, 5786.120, 11377.320]
    
    # k=7
    recall_k7 = [0.807, 0.909, 0.967, 0.991, 0.996, 0.999, 1.000, 1.000]
    blocks_k7 = [489.300, 703.500, 1057.400, 1743.420, 2431.960, 3838.760, 7419.400, 14603.820]

    # Plot results
    plt.plot(recall_vanilla, blocks_vanilla, marker='s', color="black", linewidth=2, label="vanilla pgvector")
    plt.plot(recall_k1, blocks_k1, marker='D', color="tab:red", label="w/patch(k=1)")
    plt.plot(recall_k3, blocks_k3, marker='v', color="tab:orange", label="w/patch(k=3)")
    plt.plot(recall_k5, blocks_k5, marker='^', color="tab:green", label="w/patch(k=5)")
    plt.plot(recall_k7, blocks_k7, marker='o', color="tab:blue", label="w/patch(k=7)")

    plt.xlabel("Recall")
    plt.ylabel("#Blocks")
    plt.yscale("log")
    plt.title("Recall-#Blocks tradeoff (SIFT1M,10-NN,m=24,ef_construction=200)")
    plt.legend()
    plt.grid(True, which="both")

    # Save to file
    plt.savefig("sift1m_pq_recall_blocks_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_results()

