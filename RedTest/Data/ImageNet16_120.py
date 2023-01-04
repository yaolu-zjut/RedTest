from xautodl.datasets import get_datasets
import torch
from args import args

class ImageNet16_120:

    def __init__(self):
        super(ImageNet16_120, self).__init__()

        train_data, test_data, xshape, class_num = get_datasets('ImageNet16-120', '/public/MountData/dataset/ImageNet16', 0)

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)


        self.val_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

