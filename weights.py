import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.nn import Conv2d
from torchvision.models import resnet34, resnet50
resnet = resnet34(pretrained=True)

for name, m in resnet.named_modules():
    if isinstance(m, Conv2d) and m.weight.shape[2:] == (3, 3):
        plt.clf()
        weight = m.weight.detach().numpy().ravel()
        sns.distplot(weight, color='r')
        std = np.sqrt((weight**2).sum() / (weight.size - 1))
        print(f"std weights {std}")
        # print(f"4 std weights prob {(np.abs(weight) <= 4*std).sum() / weight.size}")

        weight = m.weight.detach().numpy()
        weight = (weight[:, :, 0, :].sum(-1) + weight[:, :, 1, :].sum(-1) + weight[:, :, 2, :].sum(-1)) * 1 / 4
        weight = weight.ravel()
        sns.distplot(weight, color='b')
        std = np.sqrt((weight**2).sum() / (weight.size - 1))
        print(f"winograd std weights {std}")
        print(f"4 std winograd prob {(np.abs(weight) <= 4*std).sum() / weight.size}")

        plt.show()
        plt.waitforbuttonpress()
