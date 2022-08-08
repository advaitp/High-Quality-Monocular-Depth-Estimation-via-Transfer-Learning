import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
from zipfile import ZipFile
from sklearn.utils import shuffle
from math import exp
import torch.nn.functional as F

def gaussian(window_size, sigma) :
	gauss = torch.tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()

def createWindow(window_size, channel=1) :
	_1dWindow = gaussian(window_size, 1.5).unsqueeze(1)
	_2dWindow = _1dWindow.mm(_1dWindow.t()).float().unsqueeze(0).unsqueeze(0)
	window = _2dWindow.expand(channel, 1, window_size, window_size)

	return window

def ssim(img1, img2, window_size=11, val_range=255) :
	print(img1.size())
	(_ , channel, height, width) = img1.size()
	window = createWindow(window_size, channel).cuda()
	L = val_range

	padd = 0
	mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
	mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

	mu1sq = mu1**2
	mu2sq = mu2**2
	mu12 = mu1*mu2

	sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1sq
	sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2sq
	sigma12 =  F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu12

	C1 = (0.01 * L) ** 2
	C2 = (0.03 * L) ** 2

	numerator1 = 2 * mu12 + C1
	numerator2 = 2 * sigma12 + C2
	denominator1 = mu1sq + mu2sq + C1
	denominator2 = sigma1_sq + sigma2_sq + C2
	ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

	return ssim_score.mean()