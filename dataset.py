from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# print(len(dataset))
# print(dataset)
# print(data_loader)
for i, data in enumerate(dataset):
	print(i)
	print(data['A_paths'])
	print(data['B_paths'])
	print(data['C_paths'])
