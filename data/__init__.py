import os

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from data.dataset import SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import MarketDataset
from data.dataset import CMGDataset
from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import RandomIdentitySampler
from data.sampler import NormTripletSampler


def collate_fn(batch):  # img, label, cam_id, img_path, img_id, bbox
    new_imgs, new_labels, new_cam_ids, new_paths, new_img_ids = [], [], [], [], []
    #print(f"collate_fn received batch size: {len(batch)}")
    #print(f"batch content example: {batch[0] if len(batch)>0 else 'empty batch'}")
    for img, label, cam_id, path, img_id, bboxes in batch:
        for bbox in bboxes:
            # bbox: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, bbox)
            cropped = img.crop((x1, y1, x2, y2))  # PIL Image 裁剪
            #cropped.show()
            # 同样的 transform 重新应用
            if hasattr(img, 'transform') and callable(img.transform):
                cropped = img.transform(cropped)
                #print(f"has transform:{type(cropped)}")
            #else :
                #print("no")
            
            #cropped.show()
            new_imgs.append(cropped)
            new_labels.append(label)
            new_cam_ids.append(cam_id)
            new_paths.append(path)
            new_img_ids.append(img_id)
    imgs_tensor = torch.stack(new_imgs, dim=0)
    #print(f"imgs_tensor:{imgs_tensor.shape}")
    labels_tensor = torch.stack(new_labels, dim=0)
    #print(f"labels_tensor:{labels_tensor.shape}")
    cam_ids_tensor = torch.stack(new_cam_ids, dim=0)
    img_ids_tensor = torch.stack(new_img_ids, dim=0)

    return imgs_tensor, labels_tensor, cam_ids_tensor, new_paths, img_ids_tensor

#构建dataloader
def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=2):                 
    # data pre-processing
    t =[T.Resize(image_size)]

    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if random_erase:
        t.append(T.RandomErasing())
        # t.append(Jigsaw())

    transform = T.Compose(t)

    # dataset
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform)
    elif dataset == 'CMG':
        train_dataset = CMGDataset(root, mode='train', transform=transform)
        print("Train dataset size:", len(train_dataset))#0

    # sampler
    assert sample_method in ['random', 'identity_uniform', 'identity_random', 'norm_triplet']           #四种采样方式
    if sample_method == 'identity_uniform':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_random':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=False,
                              collate_fn=collate_fn, num_workers=num_workers)   #torch.utils.data.DataLoader在使用 sampler=参数时，要求你传入的 sampler 类（对象）必须重写__iter__,__len()__
    #print("Train dataset size:", len(train_dataset))
    return train_loader


def get_test_loader(dataset, root, batch_size, image_size, num_workers=2):
    # transform，与训练时不同，没有数据增强了
    transform=T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform)
        query_dataset = RegDBDataset(root, mode='query', transform=transform)
    elif dataset == 'market':
        gallery_dataset = MarketDataset(root, mode='gallery', transform=transform)
        query_dataset = MarketDataset(root, mode='query', transform=transform)
    elif dataset == 'CMG':
        gallery_dataset = CMGDataset(root, mode='gallery', transform=transform)
        query_dataset = CMGDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=False,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)          

    return gallery_loader, query_loader
