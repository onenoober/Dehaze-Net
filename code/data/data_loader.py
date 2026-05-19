import os, random
from pathlib import Path
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import rotate, crop
from torchvision.transforms import ToTensor, RandomCrop


IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_image_files(path):
    return sorted(
        name for name in os.listdir(path)
        if Path(name).suffix.lower() in IMAGE_SUFFIXES
    )


def resolve_pair_dirs(split_path):
    split_path = Path(split_path)
    hazy_path = resolve_first_existing_dir(split_path, ('hazy', 'haze'), 'hazy')
    clear_path = resolve_first_existing_dir(split_path, ('clear', 'gt', 'GT'), 'clear')
    return str(hazy_path), str(clear_path)


def resolve_first_existing_dir(root, names, role):
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        'No {} directory found under {}. Tried: {}'.format(
            role, root, ', '.join(str(root / name) for name in names)
        )
    )


def find_clear_image(clear_path, hazy_image_name):
    hazy_stem = Path(hazy_image_name).stem
    stems = [hazy_stem]
    prefix_stem = hazy_stem.split('_')[0]
    if prefix_stem not in stems:
        stems.append(prefix_stem)

    for stem in stems:
        for suffix in ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'):
            candidate = Path(clear_path) / f'{stem}{suffix}'
            if candidate.exists():
                return str(candidate)

    raise FileNotFoundError(f'No clear image found for {hazy_image_name} in {clear_path}')


def validate_dataset_lists(hazy_path, clear_path, hazy_image_list, clear_image_list):
    if len(hazy_image_list) == 0:
        raise FileNotFoundError(f'No hazy images found in {hazy_path}')
    if len(clear_image_list) == 0:
        raise FileNotFoundError(f'No clear images found in {clear_path}')


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, patch_size=256):
        super(TrainDataset, self).__init__()
        if patch_size <= 0:
            raise ValueError('patch_size must be positive')
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.patch_size = patch_size
        self.hazy_image_list = list_image_files(hazy_path)
        self.clear_image_list = list_image_files(clear_path)
        validate_dataset_lists(hazy_path, clear_path, self.hazy_image_list, self.clear_image_list)

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = find_clear_image(self.clear_path, hazy_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        crop_params = RandomCrop.get_params(hazy, [self.patch_size, self.patch_size])
        rotate_params = random.randint(0, 3) * 90

        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)

        hazy = rotate(hazy, rotate_params)
        clear = rotate(clear, rotate_params)

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = list_image_files(hazy_path)
        self.clear_image_list = list_image_files(clear_path)
        validate_dataset_lists(hazy_path, clear_path, self.hazy_image_list, self.clear_image_list)

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = find_clear_image(self.clear_path, hazy_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = list_image_files(hazy_path)
        self.clear_image_list = list_image_files(clear_path)
        validate_dataset_lists(hazy_path, clear_path, self.hazy_image_list, self.clear_image_list)

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = find_clear_image(self.clear_path, hazy_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}

    def __len__(self):
        return len(self.hazy_image_list)
