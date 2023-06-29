import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 8, 27, 64, 125]

# Create the figure and axes
fig, ax = plt.subplots()

# Plot the data
line1, = ax.plot(x, y1, 'r', label='y1')
line2, = ax.plot(x, y2, 'g', label='y2')

# Create the interactive legend plugin
interactive_legend = plugins.InteractiveLegendPlugin([line1, line2], ['y1', 'y2'])
plugins.connect(fig, interactive_legend)

# Convert the plot to an interactive HTML representation
html_plot = mpld3.fig_to_html(fig)

# Save the interactive plot as an HTML file
with open('interactive_plot.html', 'w') as f:
    f.write(html_plot)

plt.show()
