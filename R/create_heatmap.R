#!/usr/bin/env Rscript

# Create Cluster-Feature Heatmap with R
# Usage: Rscript create_heatmap.R <input_csv> <output_png> <n_features> <n_clusters>

suppressMessages({
  library(ComplexHeatmap)
  library(RColorBrewer)
  library(circlize)
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

# Calculate cell height in mm (for ComplexHeatmap)
# Scale based on number of features: more features = smaller cells
if (n_features <= 20) {
  cell_height_mm <- 8
} else if (n_features <= 50) {
  cell_height_mm <- 6
} else if (n_features <= 100) {
  cell_height_mm <- 4
} else {
  cell_height_mm <- max(3, 600 / n_features)  # Scale inversely, minimum 3mm
}

# Font size = 90% of cell height (10% smaller)
# Convert mm to points: 1mm ≈ 2.83465 points
y_fontsize <- round(cell_height_mm * 2.83465 * 0.9)
y_fontsize <- max(8, min(20, y_fontsize))  # Clamp between 8 and 20

# Calculate figure dimensions in mm
# Width: heatmap body + row dendrogram + labels + margins
base_width_mm <- max(200, n_clusters * 12)  # Base width for heatmap body
dendro_width_mm <- 50  # Width for row dendrogram (left)
label_width_mm <- 120  # Width for feature labels on right
margin_width_mm <- 50  # Left/right margins
fig_width_mm <- base_width_mm + dendro_width_mm + label_width_mm + margin_width_mm

# Height: heatmap body + column dendrogram + title + legend + margins
base_height_mm <- n_features * cell_height_mm
dendro_height_mm <- 40  # Height for column dendrogram (top)
title_height_mm <- 20  # Height for title
legend_height_mm <- 50  # Height for legend at bottom
margin_height_mm <- 50  # Top/bottom margins
fig_height_mm <- base_height_mm + dendro_height_mm + title_height_mm + legend_height_mm + margin_height_mm

# Create PNG with proper dimensions (convert mm to pixels at 300 DPI)
png(output_png, width = fig_width_mm * 300 / 25.4, height = fig_height_mm * 300 / 25.4, res = 300, units = "px")

# Create color mapping
# Use 100 breaks to match 100 colors
col_fun <- colorRamp2(breaks = seq(-3, 3, length.out = 100), 
                      colors = colorRampPalette(rev(brewer.pal(n = 11, name = "RdYlBu")))(100))

# Create heatmap with ComplexHeatmap
ht <- Heatmap(
  as.matrix(heatmap_data),
  name = "Z-score",
  col = col_fun,
  cluster_rows = TRUE,
  cluster_columns = TRUE,
  clustering_method_rows = "ward.D2",
  clustering_method_columns = "ward.D2",
  clustering_distance_rows = "euclidean",
  clustering_distance_columns = "euclidean",
  row_names_gp = gpar(fontsize = y_fontsize),
  column_names_gp = gpar(fontsize = 16),
  column_names_rot = 0,  # Column names upright (not rotated)
  row_names_side = "right",  # Show row names on the right
  row_dend_side = "left",  # Show row dendrogram on the left
  column_dend_side = "top",  # Show column dendrogram on top
  row_dend_width = unit(dendro_width_mm, "mm"),
  column_dend_height = unit(dendro_height_mm, "mm"),
  row_names_max_width = unit(label_width_mm, "mm"),
  heatmap_legend_param = list(
    title = "Z-score",
    title_position = "topcenter",
    legend_direction = "horizontal",
    legend_width = unit(8, "cm"),
    at = c(-3, -2, -1, 0, 1, 2, 3),
    labels = c("-3", "-2", "-1", "0", "1", "2", "3")
  ),
  cell_fun = function(j, i, x, y, width, height, fill) {
    # Optional: add borders
    grid.rect(x = x, y = y, width = width, height = height, 
              gp = gpar(fill = fill, col = "gray", lwd = 0.5))
  }
)

# Draw heatmap
draw(
  ht,
  column_title = paste0("Cluster-Feature Heatmap (Row-wise Z-score)\n", 
                         n_features, " Features × ", n_clusters, " Clusters"),
  column_title_gp = gpar(fontsize = 14, fontface = "bold"),
  heatmap_legend_side = "bottom",
  annotation_legend_side = "bottom",
  padding = unit(c(margin_height_mm, margin_width_mm, legend_height_mm, margin_width_mm), "mm")
)

dev.off()

message("Heatmap saved to: ", output_png)
