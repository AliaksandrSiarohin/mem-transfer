import matplotlib
matplotlib.use('agg')
import pylab as plt
import os
from PIL import Image
import numpy as np
import sys
good = sys.argv[2]
if not os.path.exists(good):
 	os.makedirs(good)
	
bad = sys.argv[3]
if not os.path.exists(bad):
 	os.makedirs(bad)
	
for i,line in enumerate(open(sys.argv[1])):
	print i
	if i == 0:
		continue
	in_path, style_path, out_path, in_score, out_score = line.split(',')
	in_score = float(in_score)
	out_score = float(out_score)	
		
	if out_score > in_score:
		folder = good
	else:
		folder = bad
	in_img = np.array(Image.open(in_path))
	out_img = np.array(Image.open(out_path))
	style_img = np.array(Image.open(style_path))
	
	plt.subplot('131')
	plt.imshow(in_img)
	plt.axis('off')
	plt.title(in_score)
		
	plt.subplot('132')
	plt.imshow(style_img)
	plt.axis('off')
	plt.title('style')

	plt.subplot('133')
	plt.imshow(out_img)
	plt.axis('off')
	plt.title(out_score)

	plt.savefig(os.path.join(folder, in_path.strip('/').split('/')[-1] + '_' + style_path .strip('/').split('/')[-1]+ '.png'))	
		
