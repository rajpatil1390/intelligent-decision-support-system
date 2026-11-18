# Clustering and customer segmentation

Clustering groups similar records without a target column, for example to segment customers by
behaviour. Because there is no ground truth, success is judged by whether the segments are distinct,
stable and actionable for the business, not only by a score.

Scale numeric features before clustering, otherwise columns with large ranges (income) dominate those
with small ranges (number of purchases). One-hot encode categories and keep the number of features
modest; irrelevant columns blur the clusters.

K-means is fast and works well for round, similarly sized groups. Gaussian mixture models allow
elliptical clusters and soft membership. Agglomerative (hierarchical) clustering handles irregular shapes
but needs memory proportional to the square of the number of rows, so fit it on a sample for larger
datasets. Try several numbers of clusters, typically 2 to 10.

Internal metrics help choose between options: the silhouette score (higher is better, from -1 to 1)
measures how well separated the clusters are, and the Davies-Bouldin index (lower is better) measures
how compact they are. If a known label exists, the adjusted Rand index or normalised mutual information
show how well clusters recover it.

Finally, profile each cluster in business terms: size, average values of key columns and how it differs
from the overall population. A segment is only useful if someone can name it and decide what to do
differently for it.
