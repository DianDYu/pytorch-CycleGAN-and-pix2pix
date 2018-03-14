import torch
import torch.nn as nn

class GramMatrix(nn.Module):
	
	def forward(self, input):
		a, b, c, d = input.size()
		# a: batch size
		# b: number of feature maps
		# (c, d) = dimensions of a feature map (N=c*d)
		features = input.view(a * b, c * d)
		G = torch.mm(features, features.t())
		return G.div(a * b * c * d)

		