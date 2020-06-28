from torchvision import models


attrs = dir(models)

for module in dir(models):

    print(module)
