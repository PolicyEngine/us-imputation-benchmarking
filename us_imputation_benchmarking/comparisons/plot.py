import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_comparison(loss_comparison_df, quantiles):
    percentiles = [str(int(q * 100)) for q in quantiles]
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=loss_comparison_df, x="Percentile", y="Loss", hue="Method", dodge=True)
    plt.xlabel("Percentiles", fontsize=12)
    plt.ylabel("Average Test Quantile Loss", fontsize=12)
    plt.title("Test Loss Across Quantiles for Different Imputation Methods", fontsize=14)
    plt.legend(title="Method")
    ax.set_xticklabels(percentiles)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()