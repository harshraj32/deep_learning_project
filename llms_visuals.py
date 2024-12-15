import glob
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tabulate import tabulate


def visualize_results():
    # Find all result files
    result_files = glob.glob("*_results.json")

    # Collect results
    model_results = {}
    for file in result_files:
        model_name = file.replace("_results.json", "")
        with open(file, "r") as f:
            data = json.load(f)
            model_results[model_name] = data["average_metrics"]["f1_score"]

    # Create DataFrame for visualization
    df = pd.DataFrame(
        {"Model": list(model_results.keys()), "F1 Score": list(model_results.values())}
    )

    # Create table
    print("\nModel F1 Scores:")
    print(tabulate(df, headers="keys", tablefmt="pretty", floatfmt=".4f"))

    # Create bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="F1 Score")
    plt.title("Model Comparison - F1 Scores")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("model_comparison_f1.png")
    plt.close()


if __name__ == "__main__":
    visualize_results()
