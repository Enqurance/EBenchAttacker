from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms


class Harmful(Dataset):
    def __init__(self, 
                 images_path='./resources/harmful_images/',
                 num_images=None
                 ):
        self.images = os.listdir(images_path)
        self.images.sort()
        if num_images is not None:
            self.images = self.images[:num_images]
        self.images_path = images_path
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        name = self.images[item]
        x = Image.open(os.path.join(self.images_path, name))
        return self.transforms(x), name


def get_Harmful_loader(batch_size=64,
                      num_workers=2,
                      pin_memory=True,
                      shuffle=False,
                      **kwargs,
                      ):
    
    set = Harmful(**kwargs)
    loader = DataLoader(set, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                        shuffle=shuffle)
    return loader
