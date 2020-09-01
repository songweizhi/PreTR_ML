import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


# seq_1 = 'atatatatata'
# seq_2 = 'agagagaGGGGAAA'
# seq_3 = 'ATGCatgcATGC'
#
#
# def get_gc_content(seq):
#     seq = seq.uppercase
#     gc_content = 0
#     return gc_content




X = np.array([[10,10],[2,2],[3,3],[2,2],[1,1],[2,2],[3,3],[2,2],[1,1],[77,77],[45,45],[89,89],[67,67],[97,97],[55,55],[76,76],[87,87]])
labels = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,labels)
print(clf.predict([20,20]))
w = clf.coef_[0]
print(w)


a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = labels)
plt.legend()
#plt.show()


