import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter

from model import DepthModel
from loss import ssim
from data import getTrainTestData
import matplotlib.pyplot as plt

def normalize(depth, maxDepth=1000.0) : 
    return maxDepth / depth

def main() :
	bs = 1
	epochs = 1
	lr = 0.0001
	prefix = 'densenet_' + str(bs)

	model = DepthModel().cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
	trainloader, testloader = getTrainTestData(bs)
	print('Data Loaded')
	l1_criterion = nn.L1Loss()

	# Logging
	writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, lr, epochs, bs), flush_secs=30)
	totalloss = []
	
	# for epoch in range(epochs) :
	# 	print(f'Epoch Number {epoch}')
	# 	model.train()

	# 	for i, sample in enumerate(trainloader) :
	# 		image = sample['image'].cuda()
	# 		depth = sample['depth'].cuda()

	# 		depthnorm = normalize(depth)

	# 		out = model(image)
	# 		l_depth = l1_criterion(depthnorm, out)
	# 		l_ssim = torch.clamp((1 - ssim(out, depthnorm, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

	# 		loss = (1.0 * l_ssim) + (0.1 * l_depth)

	# 		# Update step
	# 		# losses.update(loss.data.item(), image.size(0))
	# 		loss.backward()
	# 		optimizer.step()

	# 	print("Epoch {}\n Current loss {}\n".format(epoch,loss.item()))
	# 	totalloss.append(loss.item())

	# torch.save(model.state_dict(), "model.pt")
	# print("Model Saved Successfully")

	device = torch.device("cuda")
	pickle.load = partial(pickle.load, encoding="latin1")
	pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
	model = torch.load('nyu.h5', map_location=lambda storage, loc: storage, pickle_module=pickle)

	tester = testloader[0]
	image = tester['image'].cuda()
	depth = tester['depth'].cuda()
	depthnorm = normalize(depth)

	out = model(image)
	plt.imshow(out)
	plt.show()

if __name__ == "__main__" :
	main()	


