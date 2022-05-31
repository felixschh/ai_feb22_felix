import matplotlib.pyplot as plt
import numpy as np
import data_handler as dh
from model import Classify
import torch.nn.functional as F


trainset, testset, trainloader, testloader = dh.get_data()
model = Classify(784)


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.show()





dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]
img = img.resize_(img.size()[0], 784)
print(img.shape)

logits = model(img) # ps stands for probabilities: your model should return values between 0 and 1
# that sums to 1. A softmax does this job!

ps = F.softmax(logits, dim=1)

# Plot the image and probabilities
view_classify(img, ps, version='Fashion')

