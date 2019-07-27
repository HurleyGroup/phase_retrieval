import shutil, glob, os, re
import matplotlib.pyplot as plt
from itertools import imap
import torch as tc
import numpy as np
import urllib, png

PR_PATH = '/data2/adyotagupta/APS/190224_DCS/Data/Registered_Impact_PR/'
XPCI_PATH = '/data2/adyotagupta/APS/190224_DCS/Data/Registered_Impact/'
SHOT_NUM = re.compile('\d{2}-\d{1}-\d{3}')
idx_finder = re.compile('(?<=_)\d{1}')
SCALE, MU= 2.5E-6, .0005 # sets scale of 2.5 um per pixel
cuda0 = tc.device('cuda')

loader = lambda f: np.vstack( imap( np.uint16, png.Reader(file=urllib.urlopen(f)).asDirect()[2] )).astype(float)
def saver(fname, arr):
	with open(fname, 'wb') as f:
        	png.Writer(width=1024, height=1024, bitdepth=16, greyscale=True).write(f, arr.tolist())

for s in glob.glob(PR_PATH+'*'):
	# Create the new directory, or overwrite it if it exists. Handles race conditions.
	print '\nWorking Directory: ', s
	if os.path.isdir(s):
		shutil.rmtree(s)
	try:
		os.makedirs(s)
	except OSError as exc: #Handles Race Conditions
		if exc.errno != errno.EXIST:
			raise

	# for each shot, retrieve paths of images, sort based on time frame, and then load the images
	print 'Retrieving Images...'
	file_list = np.asarray(glob.glob(XPCI_PATH + re.findall(SHOT_NUM, s)[0] + '/*'))
	idx = np.asarray([int(re.findall(idx_finder,f)[0]) for f in file_list])
	file_list[idx] = np.copy(file_list)
	images = tc.tensor([loader(f) for f in file_list], device=cuda0, dtype=tc.float64)
	NUM_IM = images.shape[0]

	# Perform phase retrieval for each batch of images
	# Constants and normalize k-space using norm
	print 'Performing Phase Retrieval...'
	DELTA = 1.3E-6
	PI, LAMBDA = 3.141592654, 5.8E-11 / SCALE
	L, MU = 0.8 / SCALE , 500. * SCALE 
	images_fft = tc.rfft(images, 2, onesided=False)
	
	# Set up in plane wave propagator
	kpxx = tc.cat([tc.arange(0, images_fft.shape[1]/2, dtype=tc.float64, device=cuda0), tc.flip(tc.arange(0, images_fft.shape[2]/2, dtype=tc.float64, device=cuda0),[0])]) \
		.repeat(images_fft.shape[2],1).flatten() * (images_fft.shape[1]/2. -1.)**-1
	kpyy = tc.flip(tc.cat([tc.arange(0, images_fft.shape[2]/2, dtype=tc.float64, device=cuda0), tc.flip(tc.arange(0, images_fft.shape[2]/2, dtype=tc.float64, device=cuda0),[0])]) \
		.repeat(images_fft.shape[1],1).transpose(0,1), [0]) * (images_fft.shape[1]/2. -1.)**-1
	kp = tc.norm(tc.stack([kpxx.flatten(), kpyy.flatten()]).transpose(0,1), dim=1).view(images_fft.shape[1], images_fft.shape[2])
	kp = kp.repeat(NUM_IM*2, 1, 1, 1).view(NUM_IM,2,images_fft.shape[1], images_fft.shape[2]).permute(0,2,3,1) 

	# obtain absorption image and make it a 16 bit array
	absorp_fft = images_fft / ((L*DELTA * (kp**2) )/MU + 1.)
	absorp =  tc.ifft(absorp_fft,2)[:,:,:,0]	# Takes real part
	mins, maxs = tc.min(absorp), tc.max(absorp)
	absorp = (65535.*(absorp-mins)/(maxs-mins)).int().cpu().numpy()

	# Save all images to the correct folder
	print 'Saving Images...'
	[saver(s + '/' + str(i) + '.png', absorp[i]) for i in range(NUM_IM)]


print '\n\nDone!\n\n'
#
