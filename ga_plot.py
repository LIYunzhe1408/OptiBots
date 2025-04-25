import matplotlib.pyplot as plt

# Raw data
data = [
    (4543.333333333334, -70, -140),
    (5189.999999999999, -70, -140),
    (7476.666666666667, -60, -130),
    (7476.666666666667, -60, -130),
    (7816.666666666667, -60, -130),
    (7970.000000000001, -60, -130),
    (8460.0, -60, -130),
    (8460.0, -60, -130),
    (8460.0, -60, -130),
    (8460.0, -60, -130),
    (8716.666666666666, -60, -130),
    (8716.666666666666, -60, -130),
    (8806.666666666666, -60, -130),
    (8806.666666666666, -60, -130),
    (8856.666666666666, -60, -130),
    (8856.666666666666, -60, -130),
    (8856.666666666666, -60, -130),
    (8856.666666666666, -60, -130),
    (8883.333333333334, -60, -130),
    (8883.333333333334, -60, -130),
    (8983.333333333334, -60, -130),
    (8983.333333333334, -60, -130),
    (8983.333333333334, -60, -130)
]

# Convert to percentage of reachable space
percent_reachable = [x[0] / 100 for x in data]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(percent_reachable, marker='o', linestyle='-', color='teal')
plt.title("Genetic Algorithm - Reachable Space per Generation")
plt.xlabel("Generation")
plt.ylabel("Reachable Space (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
# Save the plot to a file
plt.savefig('/home/mscsim/Capstone2025/TimorExamples/ga_plot_output.png')