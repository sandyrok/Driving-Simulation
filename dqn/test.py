import gym
from matplotlib import pylab as plt
import numpy as np
import numpy as np
import imageio as im
import matplotlib.pylab as plt
from skimage.color import rgb2gray
from sklearn.linear_model import LinearRegression
import cv2
height = width = 96
block= 11



def compute_gradf(f1,f2,height,width):
	
	fx_diff =  (np.diff(f1,axis = 0) + np.diff(f2, axis = 0) )/ 2
	for i in range(width-1):
		fx_diff[:,i] = (fx_diff[:,i] + fx_diff[:,i + 1])/2
	fx = np.vstack((fx_diff, [0.]*width))
	
	fy_diff = ( np.diff(f1,axis = 1) + np.diff(f2, axis = 1))/2
	for i in range(height - 1):
		fy_diff[i,:] = (fy_diff[i,:] + fy_diff[i+1,:])/2
	fy = np.hstack((fy_diff,np.zeros((height,1))))
	
	ft = np.zeros_like(fy)
	ft_diff = f2 - f1
	for i in range(height):
		for j in range(width):
			ct = 1
			val = ft_diff[i,j]
			if  i + 1 < height :
				ct += 1
				val += ft_diff[i+1,j]
				if j + 1 < width:
					ct += 1
					val += ft_diff[i+1, j+1]
			if j + 1 < width:
				ct += 1
				val += ft_diff[i,j+1]
			ft[i,j] = val / ct
	
	return fx, fy, ft

def lucas_kanade(f1, f2,height,width):

	fx, fy, ft = compute_gradf(f1,f2,height,width)
	a = np.zeros((height,width))
	b = np.zeros((height,width))
	for i in range(0,height,block):
		i_1 = min(height, i + block)
		for j in range(0,width,block):
			j_1 = min(width, j + block)
			x = fx[i:i_1,j:j_1].ravel().reshape(-1,1)
			y = fy[i:i_1,j:j_1].ravel().reshape(-1,1)
			t = ft[i:i_1,j:j_1].ravel().reshape(-1,1)		
			X = np.hstack((x,y))
			Y = -t
			flow = LinearRegression().fit(X, Y).coef_
			a[i:i_1,j:j_1] = flow[0][0]
			b[i:i_1,j:j_1] = flow[0][1]
	return a,b


ls = []
env = gym.make('CarRacing-v0')
for i_episode in range(1):
   
    img = env.reset()
    for t in range(100):
        #env.render()
        #print(observation)
        ls.append(img)
        action = env.action_space.sample()
        img, reward, done, info = env.step(np.array([0,1,0]))
        """
        img = img.astype(np.int)
        plt.imshow(img)
        plt.show()
        """
        #print(type(observation),observation.dtype)
env.close()

for i in range(100):
		f1 = rgb2gray(ls[i])
		f2 = rgb2gray(ls[i+1])
		a,b = lucas_kanade(f1,f2, height,width)
		a,b = -a,-b
		b += np.arange(width,dtype = np.float32)
		a += np.arange(height,dtype = np.float32)[:,np.newaxis]
		a = a.astype(np.float32)
		b = b.astype(np.float32)
		pr = cv2.remap(ls[i], b, a, cv2.INTER_LINEAR)
		plt.imshow(pr,cmap = 'gray')
		plt.savefig('figure' + str(i+1) + '.png')





