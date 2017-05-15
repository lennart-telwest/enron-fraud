# Linear Regression
## R squared
### Definition
How much of the change in my output is explained by the change in my input.
r² ranges between 0 < r² < 1 – the lower r² the less of the output is explained by change in the input.

## Classification vs. Linear Regression

| property | supervised classification | regression |
| -------- | ------------------------- | ---------- | 
| output type | discrete (class labels) | continuous (number) |
| what are you trying to find? | decision boundary | best fit line |
| evaluation | accuracy | sum of squared error or r² |

## Outliers handling
1. Train
2. Remove points with greatest residual error. (~10%)
3. Re-Train

# Clustering
## K-Means
### Definition
Will place n centroids randomly within a dataset and relocate until the distance to the nearest observations is minimized. Each observation is associated to the nearest centroid. 

More formally: 
"The K-means algorithm involves randomly selecting K initial centroids where K is a user defined number of desired clusters. Each point is then assigned to a closest centroid and the collection of points close to a centroid form a cluster." - [hypertextbookshop](http://www.hypertextbookshop.com/dataminingbook/public_version/contents/chapters/chapter004/section002/blue/page001.html)

### Limitations:
1. When run on the same dataset multiple times, the algorithm might come up with a different segmentation, depending on the initial (random) locations of the centroids. K-Means is a [hill-climbing algorithm](https://en.wikipedia.org/wiki/Hill_climbing).
