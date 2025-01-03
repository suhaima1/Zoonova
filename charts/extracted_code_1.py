import matplotlib.pyplot as plt

# Sample data
timestamps = ['1w', '1m', '3m', '6m', '9m', '1y']
alpha_probabilities = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75]  # Example probabilities
alpha_levels = ["+3%", "+3%", "+3%", "+10%", "+10%", "+10%"]

# Create a figure and a set of subplots.
fig, ax = plt.subplots()

# Bar chart
bars = ax.bar(timestamps, alpha_probabilities, color='blue', edgecolor='black')

# Adding text labels and customizing graph
ax.set_xlabel('Time Periods')
ax.set_ylabel('Probability of Achieving Alpha')
ax.set_title('Probability of Alpha for AAPL Stock Over Various Periods')
ax.set_ylim(0, 1)  # Probability ranges from 0 to 1

# Label with alpha levels
for bar, alpha_level in zip(bars, alpha_levels):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, alpha_level, ha='center', va='bottom')

# Save the plot as a PNG file
plt.savefig('probability_of_alpha_aapl.png')

# Show the plot
plt.show()