from training_utilities import PCA, load_data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 


feature, y = load_data("../data/training/training_set.csv")
feature_2d  = PCA(feature, n_components=2)
x_min, x_max = feature_2d[:, 0].min() - .5, feature_2d[:, 0].max() + .5
y_min, y_max = feature_2d[:, 1].min() - .5, feature_2d[:, 1].max() + .5


plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(feature_2d[:, 0], feature_2d[:, 1], c=y, edgecolor='k', s=30)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("2D visualization of the training data after PCA")
plt.xlabel("1st eigenvector")
plt.ylabel("2nd eigenvector")

plt.xticks(())
plt.yticks(())
plt.savefig("../visualization/2D_visualization_xy.png", dpi=600)

fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-140, azim=120)
feature_3d = PCA(feature, n_components=3)
ax.scatter(feature_3d[:, 0], feature_3d[:, 1], feature_3d[:, 2], c=y, edgecolor='k', s=20)
ax.set_title("3D visualization of the training data after PCA")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

fig.savefig("../visualization/3D_visualization.png", dpi=600)

plt.show()