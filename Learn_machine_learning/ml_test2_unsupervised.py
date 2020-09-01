import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

X = np.array([[1, 2],[8, 8],[1.5, 1.8],[8, 8],[1, 0.6],[9, 11],[1, 10],[2, 9],[2.5, 9.5],[8.5, 12],[5,5],[5.3, 4.8],[5.9, 6.2], [9,10], [2,8], [8,2]])

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","y.", 'b.']

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=100, linewidths=3, zorder = 10)
plt.show()

