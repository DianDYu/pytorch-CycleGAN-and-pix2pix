import torch
import torch.nn as nn

import torchvision.models as models

import copy

from gramMatrix import GramMatrix
from styleloss import StyleLoss

# desired depth layers to compute style/content losses :
# content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = models.vgg19(pretrained=True).features
cnn = cnn.cuda()

def get_style_model_and_losses(cnn, style_img, style_weight, style_layers=style_layers_default):
	cnn = copy.deepcopy(cnn)
	style_losses = []

	model = nn.Sequential()
	gram = GramMatrix()

	model = model.cuda()
	gram = gram.cuda()

	i = 1
	for layer in list(cnn):
		if isinstance(layer, nn.Conv2d):
			name = "conv_" + str(i)
			model.add_module(name, layer)

			if name in style_layers:
				# add style loss:
				target_feature = model(style_img).clone()
				target_feature_gram = gram(target_feature)
				style_loss = StyleLoss(target_feature_gram, style_weight)
				model.add_module("style_loss_" + str(i), style_loss)
				style_losses.append(style_loss)

		if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
        i += 1

    	if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  

    return model, style_losses




	