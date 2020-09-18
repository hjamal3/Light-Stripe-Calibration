import numpy as np
import math
import time

def rodrigues(r):
	# https://www2.cs.duke.edu/courses/fall13/compsci527/notes/rodrigues.pdf
	theta = np.linalg.norm(r)

	if (theta == 0):
		R = np.identity(3)
	else:
		u = (r/theta).reshape(3)
		R = np.identity(3)*math.cos(theta)+(1-math.cos(theta))*np.outer(u,u)+np.array( [ [0, -u[2], u[1] ] ,[ u[2],0, -u[0] ] ,[ -u[1],u[0],0] ] )*math.sin(theta)
	return R

def fitPlane(x,y,z):
	x_1 = x-np.mean(x)
	y_1 = y-np.mean(y)
	z_1 = z-np.mean(z)

	# Use SVD and take the first two eigenvector (the principal components)
	B = np.stack((x_1,y_1,z_1))
	U, S, VH = np.linalg.svd(B)

	# eigenvectors
	p1 = U[:,0]
	p2 = U[:,1]

	# normal vectors
	p3 = np.cross(p1,p2) # the normal vector of our plane
	p3 = p3/np.linalg.norm(p3)

	# find d from the average point
	d = -(p3[0]*np.mean(x) + p3[1]*np.mean(y) + p3[2]*np.mean(z))
	c = p3
	return c,d


# use RANSAC and PCA to compute plane equation
# A: contains data points [x,y,z]
# max_iters: max number of iterations for RANSAC
# tol: tolerance of loss function
# consensus: fraction of points required for model to be accepted
# inlier: fraction of set for inliers
# returns [a,b,c,d] -> ax + by + cz + d = 0
def computePlane(A,max_iters,tol,consensus,inlier):
	n = A.shape[0]
	x = A[:,0]
	y = A[:,1]
	z = A[:,2]
	counter = 0
	best_fit = 0

	while True:

		if counter > max_iters:
			return best_fit

		# 1. Select a random subset of the original data. Call this subset the hypothetical inliers.
		sample_idx = np.random.choice(n,size=int(n*inlier),replace=False)
		x_s = x[sample_idx]
		y_s = y[sample_idx]
		z_s = z[sample_idx]

		# 2. A model is fitted to the set of hypothetical inliers.
		c,d = fitPlane(x_s,y_s,z_s)

		# 3. All other data are then tested against the fitted model. 
		# Those points that fit the estimated model well, according to some  
		# model-specific loss function, are considered as part of the consensus set.
		# average distance
		# ax + by + cz + d = 0
		l = np.absolute(A.dot(c)+d)
		num_fit = np.sum(l<tol)

		# 4. The estimated model is reasonably good if sufficiently many points 
		# have been classified as part of the consensus set.
		if num_fit > consensus*n:
			best_fit = np.append(c,d) 

		counter += 1

