import config
import dataset_vid
from torchvision.utils import save_image

from torch.utils.data import DataLoader

args = config.config_video()

dat = dataset_vid.Dataset_vid(args=args)

print(len(dat))

loader = DataLoader(
    dataset=dat, batch_size=5, num_workers=0
)

ret = next(iter(loader))

