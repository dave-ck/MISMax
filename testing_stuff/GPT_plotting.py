# GPT generated

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_count_frequency(n):
    # Read CSV file into a pandas dataframe
    path = f"permis_counts/permis_counts_graph{n}c.csv"
    df = pd.read_csv(path)

    # Prepare the bin boundaries for the histogram
    bin_boundaries = np.linspace(df['Count'].min(), df['Count'].max(), 25)  # Creates 24 bins

    # Plot histogram of the 'Count' column
    count, bins, patches = plt.hist(df['Count'], bins=bin_boundaries, edgecolor='black', label='All graphs')

    # Adding labels and title
    plt.title(f'How many permises do {n}-vertex graphs have?')
    plt.xlabel('Permis count')
    plt.ylabel('Number of graphs with permis count')

    # Modify x-axis ticks
    x_ticks = np.round(bin_boundaries[:-1]).tolist() + [df['Count'].max()]
    plt.xticks(x_ticks, rotation='vertical')

    # Plot values on top of each bar
    for i in range(len(bins) - 1):
        plt.text(bins[i] + (bins[1] - bins[0]) / 2, count[i] * 1.02, str(int(count[i])), ha='center', va='bottom',
                 fontsize=9, rotation='vertical')

    # Add red bars for the zero values
    zero_count = len(df[df['Count'] == 0])
    if zero_count > 0:
        plt.bar(0, zero_count, width=bins[1] - bins[0], color='red', align='edge', label='Permisless graphs')
        plt.ylim(0, max(count) * 1.15)

    # Add legend
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"plots/permis_count_graph{n}c.png")
    plt.show()


def plot_divisibility(n):
    path = f"permis_counts/permis_counts_graph{n}c.csv"
    # Read CSV file into a pandas dataframe
    df = pd.read_csv(path)

    # Initialize list to store counts
    divisible_counts = []

    # Calculate counts of entries divisible by each n in range(2,25)
    for k in range(2, 25):
        divisible_counts.append(sum(df['Count'] % k == 0))

    # Plot bar chart of divisible counts
    bars = plt.bar(range(2, 25), divisible_counts, edgecolor='black')

    # Adding labels and title
    plt.title(f'Number of {n}-vertex graphs with permis count divisible by each k')
    plt.xlabel('k')
    plt.ylabel('Number of graphs with 0 mod k permises')

    # Rotate x-axis labels
    plt.xticks(range(2, 25), rotation='vertical')

    # Add labeled tick at total count length
    plt.yticks(list(plt.yticks()[0]) + [len(df['Count'])])

    # Add dashed red line at total count length
    plt.axhline(y=len(df['Count']), color='r', linestyle='--', label=f'All graphs on {n} vertices')

    # Plot values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval * 1.02, str(int(yval)), ha='center', va='bottom', fontsize=9,
                 rotation='vertical')

    # Add legend
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"plots/divisibility_plot_graph{n}c.png")

    plt.show()


# Call the function with the path to your file
# plot_histogram("permis_counts/permis_counts_graph7c.csv")
for n in range(8,9):
    plot_count_frequency(n)
    plot_divisibility(n)
