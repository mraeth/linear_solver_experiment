using DelimitedFiles, CairoMakie

# Check if the file name is passed as an argument
if length(ARGS) < 1
    error("Please provide the file name as a command-line argument")
end

# Read the file name from command-line arguments
file_name = ARGS[1]

# Read data from the provided file
data = readdlm(file_name)

# Create density plot
fig, ax, plt = heatmap(data, colormap=:viridis)

# Customize the plot
Colorbar(fig, plt)
#axis!(ax, xlabel="X", ylabel="Y", title="Density Plot")

# Save or display the figure
save("density_plot.png", fig)  # Save the plot as an image file
fig  # Display the plot

