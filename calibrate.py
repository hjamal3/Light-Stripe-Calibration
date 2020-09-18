# import modules
import numpy as np
import cv2
import time, glob, sys
import sympy as sym
from sympy import Pow, re, sqrt, Abs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from helper import rodrigues, computePlane
import skimage.restoration
from skimage import img_as_ubyte

# print CV version
print('CV Version: ' + cv2.__version__)

# load camera calibration results
# later use the opencv yaml file. right camera
# https://docs.opencv.org/3.4.10/dd/d74/tutorial_file_input_output_with_xml_yml.html
# for now hard code, couldn't figure out how to do read non OpenCV matrix in python
mtx = np.reshape([ 514.41205,     0.     ,    329.83671,
			 0.     ,  685.92876,  237.71471,
			 0.     ,     0.     ,     1.     ],(3,3))
dist = np.array([-0.350373, 0.158447, 0.000735, -0.000231, 0.000000])
length_square = 0.04 # length of side of calibration square in m
corners_shape = (6,8)

# configuration options
vertical = True # are the lasers vertical or horizontal
green = True # green or red lasers

# create empty matrix-> solving for plane equation: ax + by + cz + d = 0
Q_c_mat = np.empty((0,4))

# number of vertical corners
h = corners_shape[0] 

# number of horizontal corners
w = corners_shape[1] 

# read images
images = glob.glob('./images/*.jpg')

# if no images exit
if (len(images) == 0):
	print('Error: no photos in images folder')
	sys.exit(0)

# iterate through images
for fname in images:
	print("")
	print(fname)

	# load current image
	img_org = cv2.imread(fname,cv2.IMREAD_COLOR)

	# undistort the image -> use this for finding laser
	img_undist = cv2.undistort(img_org, mtx, dist)

	# denoise image -> for finding checkerboard
	img_denoised =  cv2.fastNlMeansDenoisingColored(img_undist,None,10,10,7,21)

	# convert to greyscale
	gray = cv2.cvtColor(img_denoised,cv2.COLOR_BGR2GRAY)

	# convert to hsv
	hsv = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HSV)

	if green:
		# use red color space to find checkerboard
		imR = img_undist[:,:,2] 

		# this mask was found to work well for green 
		mask = cv2.inRange(hsv, (50, 60, 50), (100, 255,255))

		imask = mask>0

	else: # red laser
		# use green color space to find checkerboard
		imR = img_undist[:,:,0] 

		# these masks were found to work well for red
		mask1 = cv2.inRange(hsv,(0, 80, 50), (10, 255, 255))
		mask2 = cv2.inRange(hsv,(170, 80, 50), (180, 255, 255))
		imask = np.logical_or(mask1 > 0, mask2>0)

	# use mask over the original image.
	# To do: use the mask directly?
	img_filtered = np.zeros_like(img_undist, np.uint8)
	img_filtered[imask] = img_undist[imask]

	# TO DO: Find better ways of closing line segments
	# Zhang Suen fast parallel thinning algorithm:
	laser_img = cv2.ximgproc.thinning(cv2.cvtColor(img_filtered, cv2.COLOR_RGB2GRAY))
	kernel = np.ones((100,100),np.uint8)
	laser_img = cv2.morphologyEx(laser_img, cv2.MORPH_CLOSE, kernel)
	laser_img = cv2.morphologyEx(laser_img, cv2.MORPH_CLOSE, kernel)

	# img_intensity = np.zeros_like(img_undist,np.uint8)
	# img_intensity[gray>160] = 255

	# if True:
	# 	cv2.imshow('image',img_intensity)
	# 	cv2.waitKey(0)
	# 	cv2.destroyAllWindows()


	# create object points (3D points on the calibration grid): corners (x,y) -> x right, y down
	ret, corners = cv2.findChessboardCorners(gray, corners_shape,None)

	# find the corners, findChessboardCorners, use cornerSubPix
	if not ret:
		print("No chessboard pattern found.")
		continue

	# refine corners -> termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# refine corners
	corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

	# find 3D object points
	objp = np.zeros((w*h,3), np.float32)
	objp[:,:2] = np.mgrid[0:h,0:w].T.reshape(-1,2)

	# scale with length of each square
	objp *= length_square

	# TODO: generate a list of top,mid,bot coords to do the search on rather than just one point

	# loop through it

	if not vertical:
		# pick 3 corner points in a vertical line in the middle of the checkerboard: 2D a,b,c and 3D A, B, C
		# a should be top, b middle, c bottom.
		top_coord = (0,int(w/2.0)-1)
		mid_coord = (int(h/2.0),int(w/2.0)-1)
		bot_coord = (h-1,int(w/2.0)-1)
	else:
		# pick 3 corners in a horizontal line in the middle of the checkerboard: 2D a,b,c and 3D A, B, C
		top_coord = (int(h/2.0), 0)
		mid_coord = (int(h/2.0), int(w/2.0)-1)
		bot_coord = (int(h/2.0),w-1)

	# get the index of a, b, c
	top_idx = top_coord[0]+top_coord[1]*h
	mid_idx = mid_coord[0]+mid_coord[1]*h
	bot_idx = bot_coord[0]+bot_coord[1]*h

	# get the coordinates of a, b, c
	a_coord = corners[top_idx,:,:].reshape((2))
	b_coord = corners[mid_idx,:,:].reshape((2))
	c_coord = corners[bot_idx,:,:].reshape((2))

	# find q, the laser point between a,b, and c 
	# search between a and c -> convex combination q_candidate = t*a+(1-t)*c, t in (0,1)
	laser_found = False
	for t_candidate in np.linspace(0,1,num=gray.shape[1]):
		q_candidate = (t_candidate*a_coord+(1-t_candidate)*c_coord).astype(int)
		if (laser_img[q_candidate[1],q_candidate[0]] > 200):
			# TODO: don't just take the first point. take the middle one. laser is still a few pixels wide.
			laser_found = True
			q_coord = q_candidate
			break

	# no laser found along corner points line
	if not laser_found:
		print("Failed to find laser point. Skipping picture. Chg thresholds maybe?")
		continue

	# visualize the four points a,b,c,q
	if False:
		pts = np.vstack((a_coord,b_coord,c_coord)).astype(int)
		img_circles = cv2.drawChessboardCorners(cv2.cvtColor(img_undist,cv2.COLOR_BGR2GRAY), corners_shape, corners,ret)

		# Draw a circle with blue line borders of thickness of 2 px 
		for i in range(0,pts.shape[0]):
			# fyi, CVPoint = (x,y) -> x right, y down.
			img_circles = cv2.circle(img_circles, (pts[i,0],pts[i,1]), 10, (255, 0, 0) , 2) 
		img_circles = cv2.circle(img_circles, (q_coord[0],q_coord[1]), 10, (0, 0, 0) , 2) 
		q_coord
		cv2.imshow('img',img_circles)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# convert a,b,c,q to normalized coordinates p_n -> p_u = Kp_n  (p_n = [I|0]p_c), (p_c = [R|t]p_w)
	# so p_n = inv(K)*p_u
	a = np.linalg.inv(mtx).dot(np.append(a_coord,1))
	a = a[:2]/a[2]
	b = np.linalg.inv(mtx).dot(np.append(b_coord,1))
	b = b[:2]/b[2]
	c = np.linalg.inv(mtx).dot(np.append(c_coord,1))
	c = c[:2]/c[2]
	q = np.linalg.inv(mtx).dot(np.append(q_coord,1))
	q = q[:2]/q[2]

	# compute cross ratio using pixels a, b, c
	# ((AB)/(QB))/((AC)/(QC)) = ((ab)/(qb))/((ac)/(qc))
	A = objp[top_idx,:]
	B = objp[mid_idx,:]
	C = objp[bot_idx,:]

	# q is either in between a and b, or between b and c.
	# cross ratio is a line of a, q, b, c or a, b, q, c
	q_left = False
	if np.linalg.norm(a-q) < np.linalg.norm(a-b):
		q_left = True

	# q is between a and b: cr(a,q,b,c) = cr(A,Q,B,C) = (AB/QB)/(AC/QC)
	if q_left:
		ab = np.linalg.linalg.norm(a-b)
		qb = np.linalg.linalg.norm(q-b)
		ac = np.linalg.linalg.norm(a-c) 
		qc = np.linalg.linalg.norm(q-c)
		cross_ratio = ((ab)/(qb))/((ac)/(qc))
		AB = np.linalg.norm(A-B)
		AC = np.linalg.norm(A-C)
		BC = AC-AB
		# use cross ratio + geometry
		QB = sym.symbols('QB',real=True)
		QB = sym.solve((AB/QB)/(AC/(BC+QB))-cross_ratio, QB)[0] # cross
		QC = BC + QB # geometric relationship: BC + QB = QC 
		Qx, Qy = sym.symbols('Qx, Qy')
		sol = sym.solve((sqrt(Pow(Qx - B[0],2) + Pow(Qy - B[1],2))-QB,sqrt(Pow(Qx - C[0],2) + Pow(Qy - C[1],2))-QC),(Qx,Qy))
	else: # cr(a,b,q,c) = cr(A,B,Q,C) = (AQ/BQ)*(AC/BC)
		# q is between b and c... flip the problem (swap a and c from above condition)
		aq = np.linalg.linalg.norm(q-a)
		qb = np.linalg.linalg.norm(q-b)
		ac = np.linalg.linalg.norm(a-c) 
		bc = np.linalg.linalg.norm(b-c) 

		cross_ratio = ((aq)/(qb))/((ac)/(bc))
		AC = np.linalg.norm(A-C)
		CB = np.linalg.norm(C-B)
		AB = AC-CB
		# use cross ratio + geometry
		QB = sym.symbols('QB',real=True)
		QB = sym.solve(((AB+QB)/QB)/(AC/CB)-cross_ratio, QB)[0] # cross
		QA = AB + QB # geometric relationship: BC + QB = QC 
		Qx, Qy = sym.symbols('Qx, Qy')
		sol = sym.solve((sqrt(Pow(Qx - B[0],2) + Pow(Qy - B[1],2))-QB,sqrt(Pow(Qx - A[0],2) + Pow(Qy - A[1],2))-QA),(Qx,Qy))

	if False:
		pts = np.vstack((A,B,C))
		plt.scatter(pts[:,0],pts[:,1])
		plt.show()

	# time.sleep(10)
	if (len(sol) == 0):
		print('No solution found.')
		continue
	Qx = re(sol[0][0])
	Qy = re(sol[0][1])

	# solve for Q
	Q = np.array([Qx,Qy,0])

	if False:
		pts = np.vstack((A,B,C))
		plt.scatter(pts[:,0],pts[:,1])
		plt.show()

	# make Q homog.
	Q = np.append(Q,1)

	# correspondences are column wise
	# solve PnP to get camera rotation and translation matrix
	# that transform a 3D point expressed in the object coordinate frame to the camera coordinate frame
	ret, rvec, tvec = cv2.solvePnP(objp,corners,mtx,np.zeros((5,1))) # rvec is rodrigues vector

	# put Q into camera frame using rvec, tvec
	R = rodrigues(rvec)
	T = np.vstack((np.concatenate((R,tvec),axis=1),[0,0,0,1]))

	Q_c = T.dot(Q)
	print('3D point of laser found:')
	print(Q_c[:3])

	# Add Q_c to Q_c_mat
	Q_c_mat = np.vstack((Q_c_mat,Q_c))

if (Q_c_mat.shape[0] < 3):
	print('Error: not enough laser points found to find plane eqn. Need at least 3')
	sys.exit(0)
else:
	print('Found a total of '+ str(Q_c_mat.shape[0]) + ' 3D points. Computing plane.')
	# np.save('Q', Q_c_mat.astype(np.double))
	# solve for the plane
	l = computePlane(Q_c_mat[:,:3].astype(np.double),1000,0.001,0.5,0.2)

	if type(l) == int: # return 0 if RANSAC doesn't pass
		print('Error: maximum RANSAC iterations passed, no solution.')
		sys.exit(0)

	# u, s, vh = np.linalg.svd(Q_c_mat.astype(np.double))
	# l = vh[-1,:]
	print('Plane equation of laser [a,b,c,d]: ')
	print(l)

	if True:
		fig = plt.figure()
		Q = Q_c_mat.astype(np.double)
		ax = Axes3D(fig)
		ax.scatter(Q[:,0], Q[:,1], Q[:,2])
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('Z Label')
		plt.show()