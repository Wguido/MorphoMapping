#!/usr/bin/env Rscript

# Create Cluster-Feature Heatmap with R
# Usage: Rscript create_heatmap.R <input_csv> <output_png> <n_features> <n_clusters>

suppressMessages({
  library(pheatmap)
  library(RColorBrewer)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript create_heatmap.R <input_csv> <output_png> <n_features> <n_clusters>")
}

input_csv <- args[1]
output_png <- args[2]
n_features <- as.integer(args[3])
n_clusters <- as.integer(args[4])

# Read data
heatmap_data <- read.csv(input_csv, row.names = 1, check.names = FALSE)

# Calculate font sizes based on number of features
if (n_features <= 20) {
  y_fontsize <- 28
} else if (n_features <= 50) {
  y_fontsize <- 24
} else if (n_features <= 100) {
  y_fontsize <- 20
} else {
  y_fontsize <- 16
}

# Calculate figure size
fig_width <- max(14, n_clusters * 1.5)
fig_height <- max(12, n_features * 0.3)

# Create heatmap with pheatmap
png(output_png, width = fig_width * 100, height = fig_height * 100, res = 300)

pheatmap(
  heatmap_data,
  color = colorRampPalette(rev(brewer.pal(n = 11, name = "RdYlBu")))(100),
  cluster_rows = TRUE,  # Cluster features (Y-axis)
  cluster_cols = TRUE,  # Cluster clusters (X-axis)
  clustering_method = "ward.D2",
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  show_rownames = if (n_features <= 100) TRUE else FALSE,
  show_colnames = TRUE,
  fontsize_row = y_fontsize,
  fontsize_col = 18,
  fontsize = 10,
  fontsize_number = 8,
  angle_col = 0,  # Horizontal labels
  labels_row = rownames(heatmap_data),
  labels_col = colnames(heatmap_data),
  main = paste0("Cluster-Feature Heatmap (Row-wise Z-score)\n", 
                n_features, " Features × ", n_clusters, " Clusters"),
  border_color = "gray",
  cellwidth = if (n_clusters <= 20) 20 else NA,
  cellheight = if (n_features <= 50) 20 else NA,
  legend = TRUE,
  legend_breaks = c(-3, -2, -1, 0, 1, 2, 3),
  legend_labels = c("-3", "-2", "-1", "0", "1", "2", "3"),
  breaks = seq(-3, 3, length.out = 101)
)

dev.off()

message("Heatmap saved to: ", output_png)

