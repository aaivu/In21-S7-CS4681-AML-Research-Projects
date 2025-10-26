import matplotlib.pyplot as plt

# Data from the table
levels = [1, 3, 5, 6, 7]
map_at_0_5 = [51.8, 64.4, 70.2, 71.0, 70.6]
map_at_0_7 = [15.8, 31.5, 42.2, 43.9, 43.2]
avg_map = [47.6, 60.1, 65.5, 66.8, 66.2]

# Create the plot
# A line plot with markers is often clearer than a pure scatter plot for this kind of data
plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each mAP metric
ax.plot(levels, map_at_0_5, marker='o', linestyle='-', label='mAP@0.5')
ax.plot(levels, map_at_0_7, marker='s', linestyle='-', label='mAP@0.7')
ax.plot(levels, avg_map, marker='^', linestyle='-', label='Avg. mAP')

# Highlight the best performing level (6 levels)
ax.scatter(6, 71.0, s=150, facecolors='none', edgecolors='red', linewidth=1.5, label='Best Performance (# Levels = 6)')
ax.scatter(6, 43.9, s=150, facecolors='none', edgecolors='red', linewidth=1.5)
ax.scatter(6, 66.8, s=150, facecolors='none', edgecolors='red', linewidth=1.5)


# Add titles and labels for clarity
ax.set_title('Performance vs. Number of Feature Pyramid Levels', fontsize=16)
ax.set_xlabel('# Levels', fontsize=12)
ax.set_ylabel('mAP Score', fontsize=12)

# Set x-axis to only show integer values
ax.set_xticks(levels)

# Add a legend to identify the lines
ax.legend(fontsize=10)

# Display the plot
plt.tight_layout()
plt.show()