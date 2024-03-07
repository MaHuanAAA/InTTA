import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import BertTokenizer, AutoTokenizer, DistilBertTokenizer, GPT2Tokenizer
from torchvision import datasets
from utils_box import makefig_GSAM,makefig_GSAM_box
from randaugment import RandAugment

caption_dict={
        'Waterbirds':"a photo of ",
        'CelebA':"a photo of ",
        'PACS':"an image of ",
        'VLCS':"a photo of ",
        'ImagenetA':"a photo of ",
        'ImagenetR':"a photo of ",
        'Imagenetv2':"a photo of ",
        'tinyImagenet':"a photo of ",
        'Stanforddogs': "a photo of ",
        'CUB': "a photo of",
        'oxford3pet': "a photo of ",
        'spawrious224hard': "a photo of ",
        'spawrious224': "a photo of ",
        'spawrious224medium': "a photo of ",
        'wolfdog': 'a photo containing ',
        'cameldeer': 'a photo containing ',
        'crabspider': 'a photo containing ',
        'wolfhusky': 'a photo containing '
    }

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 0            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None,meta_part=None):
        # df = pd.read_csv(metadata)
        df = metadata
        df = df[df["split"] == (self.SPLITS[split])]

        print(len(df))
        if meta_part is not None:
            gap = len(df) // meta_part['TOTAL_PATCH']
            
            if meta_part['TOTAL_PATCH'] == meta_part['CURRENT_PATCH']:
                df = df[(meta_part['CURRENT_PATCH']-1)*gap:]
            else:
                df = df[(meta_part['CURRENT_PATCH']-1)*gap:meta_part['CURRENT_PATCH']*gap]
        print(len(df))

        self.idx = list(range(len(df)))
        self.root = root
        # self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.x = df["filename"].astype(str).tolist()
        df["y"][df["y"] == -1] = 0
        df["a"][df["a"] == -1] = 0
        self.y = df["y"].tolist()
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        if self.use_anno:
            self.box_anno = [df["box1"].tolist(),df["box2"].tolist(),
                             df["box3"].tolist(),df["box4"].tolist()]
        self.transform_ = transform
        self._count_groups()
        if self.shot > 0:
            gs = [f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(self.y, self.a))]
            new_idx = []
            for g in np.unique(gs):
                num = 0
                for now_idx in self.idx:
                    if g == f'y={self.y[now_idx]},a={self.a[now_idx]}':
                        num += 1
                        if num == self.shot+1:
                            new_idx.append(now_idx)

                    if num == self.shot+1:
                        break
            self.idx = new_idx

        # if self.shot > 0:
        #     gs = [f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(self.y, self.a))]
        #     new_idx = []
        #     for g in np.unique(gs):
        #         num = 0
        #         for now_idx in self.idx:
        #             if g == f'y={self.y[now_idx]},a={self.a[now_idx]}':
        #                 num += 1
        #                 new_idx.append(now_idx)

        #             if num == self.shot:
        #                 break
        #     self.idx = new_idx


        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        # self.num_labels = len(set(self.y))
        self.num_labels = len(set(self.class_name))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels
        for i in self.idx:
            # print(self.num_attributes * self.y[i] + self.a[i])
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        if self.use_anno:
            x = self.transform(self.x[i],[self.box_anno[0][i],
                                          self.box_anno[1][i],
                                          self.box_anno[2][i],
                                          self.box_anno[3][i]])
        else:
            x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)

class Waterbirds(SubpopDataset):
    CHECKPOINT_FREQ = 1
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams=None, train_attr='yes', 
                        subsample_type=None, duplicates=None,transform=None,
                        box=False,shot=0,raw_aug=None,for_aug=None,tpt_aug=None,
                        withnosam=False,meta_part=None):
        self.use_anno = False
        root = os.path.join(data_path, "waterbirds", "waterbird_complete95_forest2water2")
        metadata = os.path.join(root, "metadata.csv")
        metadata = pd.read_csv(metadata)
        self.box = box
        self.aug_path = os.path.join(data_path, "waterbirds", "waterbird_aug")
        if box:
            self.aug_path = os.path.join(data_path, "waterbirds", "waterbird_aug_box")
        self.shot = shot
        # transform = transforms.Compose([
        #     transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        self.data_type = "images"
        self.class_name = ['a landbird','a waterbird']
        self.raw_aug_num = 0
        if raw_aug is not None:
            self.raw_aug = raw_aug['aug']
            self.raw_aug_num = raw_aug['num']
        self.for_aug_num = 0
        if for_aug is not None:
            self.for_aug = for_aug['aug']
            self.for_aug_num = for_aug['num']
            self.same = for_aug['same']
        # self.class_name = ['landbird','waterbird']
        self.tpt_aug = tpt_aug
        self.withnosam = withnosam
        self.nosam_aug = RandAugment(1, strong=True)
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates,meta_part=meta_part)

    def transform(self, image_path):
        raw_image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        # raw_image = np.array(raw_image,dtype=np.uint8)
        image_name = image_path.split('/')[1]
        save_path = os.path.join(self.aug_path,image_path.split('/')[0]) + '/'
        if self.box and not self.withnosam:
            if os.path.exists(save_path+ image_name+'box_mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'box_mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")
                # mask_back = Image.open(save_path.replace("waterbird_aug_box", "waterbird_aug")+'mask_back.jpeg').convert("RGB")  # 消融实验，记得改回来
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM_box(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM_box(raw_image,save_path,'',0.1,0.1)
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")

        elif not self.withnosam:
            if os.path.exists(save_path+ image_name+'mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM(raw_image,save_path,'',0.1,0.1)
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
        
        if self.withnosam:
            mask_for = Image.open(os.path.join(self.root, image_path)).convert("RGB")
            mask_back = mask_for
        if self.raw_aug_num > 0 or self.for_aug_num > 0:
            # mask_back_list = [self.transform_(mask_back)]
            mask_back_list = [self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]
            mask_back_list_before_trans = []
            if self.raw_aug_num > 0:
                for i in range(self.raw_aug_num-1):
                    if self.withnosam:
                        mask_back_aug = self.nosam_aug(mask_back)
                    else:
                        mask_back_aug = self.raw_aug(mask_back)

                    mask_back_list.append(self.transform_(mask_back_aug))

            if self.for_aug_num > 0:
                mask_for_list = [self.transform_(mask_for)]
                for i in range(self.for_aug_num-1):
                    mask_for_list.append(self.transform_(self.for_aug(mask_for)))

            else:
                mask_for_list = self.transform_(mask_for)
            return {'image_path':image_path,
                    'raw':self.transform_(raw_image),
                    'for':mask_for_list,
                    'back':mask_back_list}
        else:
            if self.tpt_aug is not None:
                return {'image_path':image_path,
                        'raw':self.tpt_aug(raw_image),
                        'for':self.transform_(mask_for),
                        'back':self.transform_(mask_back)}
            else:
                return {'image_path':image_path,
                        'raw':self.transform_(raw_image),
                        'for':self.transform_(mask_for),
                        'back':[self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]}
        # return self.transform_(Image.open(x).convert("RGB"))

class PACS(SubpopDataset):
    CHECKPOINT_FREQ = 1
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams=None, train_attr='yes', 
                        subsample_type=None, duplicates=None,transform=None,
                        box=False,shot=0,raw_aug=None,for_aug=None,tpt_aug=None,
                        withnosam=False,meta_part=None):
        self.use_anno = False
        root = os.path.join(data_path, "PACS")
        metadata = os.path.join(root, "metadata.csv")
        metadata = pd.read_csv(metadata)
        self.box = box
        self.aug_path = os.path.join(data_path, "PACS", "PACS_aug")
        if box:
            self.aug_path = os.path.join(data_path, "PACS", "PACS_aug_box")
        self.shot = shot

        self.data_type = "images"
        self.class_name = ['dog','elephant','giraffe','guitar','horse','house','person']

        self.raw_aug_num = 0
        if raw_aug is not None:
            self.raw_aug = raw_aug['aug']
            self.raw_aug_num = raw_aug['num']
        self.for_aug_num = 0
        if for_aug is not None:
            self.for_aug = for_aug['aug']
            self.for_aug_num = for_aug['num']
            self.same = for_aug['same']
        self.tpt_aug = tpt_aug
        self.withnosam = withnosam
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates,meta_part=meta_part)

    def transform(self, image_path):
        raw_image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        # raw_image = np.array(raw_image,dtype=np.uint8)
        image_name = image_path.split('/')[-1]
        save_path = os.path.join(self.aug_path,'/'.join(image_path.split('/')[:-1])) + '/'
        if self.box:
            if os.path.exists(save_path+ image_name+'box_mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'box_mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM_box(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM_box(raw_image,save_path,'',0.1,0.1)
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")

        elif not self.withnosam:
            if os.path.exists(save_path+ image_name+'mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM(raw_image,save_path,'',0.1,0.1)
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")

        # return {'raw':self.transform_(raw_image),
        #         'for':self.transform_(mask_for),
        #         'back':self.transform_(mask_back)}
        # return self.transform_(Image.open(x).convert("RGB"))
        if self.withnosam:
            mask_for = Image.open(os.path.join(self.root, image_path)).convert("RGB")
            mask_back = transforms.ToTensor()(mask_for).cuda() + torch.distributions.normal.Normal(0, 0.1).sample(transforms.ToTensor()(mask_for).shape).cuda()
            mask_back = transforms.ToPILImage()(mask_back)
        if self.raw_aug_num > 0 or self.for_aug_num > 0:
            # mask_back_list = [self.transform_(mask_back)]
            mask_back_list = [self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]
            mask_back_list_before_trans = []
            if self.raw_aug_num > 0:
                for i in range(self.raw_aug_num-1):
                    # r = (transforms.ToTensor()(mask_back).cuda()[0,:,:] == 0)
                    # g = (transforms.ToTensor()(mask_back).cuda()[1,:,:] == 0)
                    # b = (transforms.ToTensor()(mask_back).cuda()[2,:,:] == 0)
                    # zero_site = torch.where(torch.logical_and(b,torch.logical_and(r,g))==True)
                    # mask_back_aug = transforms.ToTensor()(mask_back).cuda() + torch.distributions.normal.Normal(0, 0.03).sample(transforms.ToTensor()(mask_back).shape).cuda()
                    # mask_back_aug[:,zero_site[0],zero_site[1]] = torch.zeros_like(mask_back_aug[:,zero_site[0],zero_site[1]]).cuda()
                    mask_back_aug = self.raw_aug(mask_back)

                    mask_back_list.append(self.transform_(mask_back_aug))

            if self.for_aug_num > 0:
                mask_for_list = [self.transform_(mask_for)]
                for i in range(self.for_aug_num-1):
                    mask_for_list.append(self.transform_(self.for_aug(mask_for)))

            else:
                mask_for_list = self.transform_(mask_for)
            return {'image_path':image_path,
                    'raw':self.transform_(raw_image),
                    'for':mask_for_list,
                    'back':mask_back_list}
        else:
            if self.tpt_aug is not None:
                return {'image_path':image_path,
                        'raw':self.tpt_aug(raw_image),
                        'for':self.transform_(mask_for),
                        'back':self.transform_(mask_back)}
            else:
                return {'image_path':image_path,
                        'raw':self.transform_(raw_image),
                        'for':self.transform_(mask_for),
                        'back':[self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]}

class domainnet(SubpopDataset):
    CHECKPOINT_FREQ = 1
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams=None, train_attr='yes', 
                        subsample_type=None, duplicates=None,transform=None,
                        box=False,shot=0,raw_aug=None,for_aug=None,tpt_aug=None,
                        withnosam=False,meta_part=None):
        self.use_anno = False
        root = os.path.join(data_path, "domainnet_v1.0")
        metadata = os.path.join(root, "metadata.csv")
        metadata = pd.read_csv(metadata)

        attr_dict = {
            'clipart':0,
            'infograph':1,
            'painting':2,
            'quickdraw':3,
            'real':4,
            'sketch':5,
        }

        metadata['a'] = metadata['a'].apply(lambda x: attr_dict[x])
        metadata['split'][metadata['split'] == 'train'] = 0
        metadata['split'][metadata['split'] == 'test'] = 2

        class_id = set(zip(metadata['y'].tolist(),metadata['category'].tolist()))
        class_id = sorted(list(class_id),key=lambda n:n[0])
        self.class_name = []
        for i in range(len(class_id)):
            if i == class_id[i][0]:
                self.class_name.append(class_id[i][1])
            else:
                assert 0
        self.box = box
        self.aug_path = os.path.join(data_path, "domainnet_v1.0", "aug")
        if box:
            self.aug_path = os.path.join(data_path, "domainnet_v1.0", "aug_box")
        self.shot = shot
        # transform = transforms.Compose([
        #     transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        self.data_type = "images"
        # self.class_name = ['a landbird','a waterbird']
        self.raw_aug_num = 0
        if raw_aug is not None:
            self.raw_aug = raw_aug['aug']
            self.raw_aug_num = raw_aug['num']
        self.for_aug_num = 0
        if for_aug is not None:
            self.for_aug = for_aug['aug']
            self.for_aug_num = for_aug['num']
            self.same = for_aug['same']
        # self.class_name = ['landbird','waterbird']
        self.tpt_aug = tpt_aug
        self.withnosam = withnosam
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates,meta_part=meta_part)

    def transform(self, image_path):
        raw_image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        # raw_image = np.array(raw_image,dtype=np.uint8)
        image_name = image_path.split('/')[-1]
        save_path = os.path.join(self.aug_path,'/'.join(image_path.split('/')[:-1])) + '/'
        if self.box:
            if os.path.exists(save_path+ image_name+'box_mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'box_mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")
                # mask_back = Image.open(save_path.replace("waterbird_aug_box", "waterbird_aug")+'mask_back.jpeg').convert("RGB")  # 消融实验，记得改回来
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM_box(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM_box(raw_image,save_path,'',0.1,0.1)
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")

        elif not self.withnosam:
            if os.path.exists(save_path+ image_name+'mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM(raw_image,save_path,'',0.1,0.1)
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
        
        if self.withnosam:
            mask_for = Image.open(os.path.join(self.root, image_path)).convert("RGB")
            mask_back = transforms.ToTensor()(mask_for).cuda() + torch.distributions.normal.Normal(0, 0.1).sample(transforms.ToTensor()(mask_for).shape).cuda()
            mask_back = transforms.ToPILImage()(mask_back)
        if self.raw_aug_num > 0 or self.for_aug_num > 0:
            # mask_back_list = [self.transform_(mask_back)]
            mask_back_list = [self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]
            mask_back_list_before_trans = []
            if self.raw_aug_num > 0:
                for i in range(self.raw_aug_num-1):
                    # r = (transforms.ToTensor()(mask_back).cuda()[0,:,:] == 0)
                    # g = (transforms.ToTensor()(mask_back).cuda()[1,:,:] == 0)
                    # b = (transforms.ToTensor()(mask_back).cuda()[2,:,:] == 0)
                    # zero_site = torch.where(torch.logical_and(b,torch.logical_and(r,g))==True)
                    # mask_back_aug = transforms.ToTensor()(mask_back).cuda() + torch.distributions.normal.Normal(0, 0.03).sample(transforms.ToTensor()(mask_back).shape).cuda()
                    # mask_back_aug[:,zero_site[0],zero_site[1]] = torch.zeros_like(mask_back_aug[:,zero_site[0],zero_site[1]]).cuda()
                    mask_back_aug = self.raw_aug(mask_back)

                    mask_back_list.append(self.transform_(mask_back_aug))

            if self.for_aug_num > 0:
                mask_for_list = [self.transform_(mask_for)]
                for i in range(self.for_aug_num-1):
                    mask_for_list.append(self.transform_(self.for_aug(mask_for)))

            else:
                mask_for_list = self.transform_(mask_for)
            return {'image_path':image_path,
                    'raw':self.transform_(raw_image),
                    'for':mask_for_list,
                    'back':mask_back_list}
        else:
            if self.tpt_aug is not None:
                return {'image_path':image_path,
                        'raw':self.tpt_aug(raw_image),
                        'for':self.transform_(mask_for),
                        'back':self.transform_(mask_back)}
            else:
                return {'image_path':image_path,
                        'raw':self.transform_(raw_image),
                        'for':self.transform_(mask_for),
                        'back':[self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]}
        # return self.transform_(Image.open(x).convert("RGB"))

class cameldeer(SubpopDataset):
    CHECKPOINT_FREQ = 1
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams=None, train_attr='yes', 
                        subsample_type=None, duplicates=None,transform=None,
                        box=False,shot=0,raw_aug=None,for_aug=None,tpt_aug=None,
                        withnosam=False,meta_part=None):
        self.use_anno = False
        root = os.path.join(data_path, "camel_deer")
        metadata = os.path.join(root, "metadata.csv")
        metadata = pd.read_csv(metadata)
        self.nosam_aug = RandAugment(1, strong=True)
        self.box = box
        self.aug_path = os.path.join(data_path, "camel_deer", "camel_deer_aug")
        if box:
            self.aug_path = os.path.join(data_path, "camel_deer", "camel_deer_aug_box")
        self.shot = shot
        # transform = transforms.Compose([
        #     transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        self.data_type = "images"
        self.class_name = ['camel','deer']
        self.raw_aug_num = 0
        if raw_aug is not None:
            self.raw_aug = raw_aug['aug']
            self.raw_aug_num = raw_aug['num']
        self.for_aug_num = 0
        if for_aug is not None:
            self.for_aug = for_aug['aug']
            self.for_aug_num = for_aug['num']
            self.same = for_aug['same']
        # self.class_name = ['landbird','waterbird']
        self.tpt_aug = tpt_aug
        self.withnosam = withnosam
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates,meta_part=meta_part)

    def transform(self, image_path):
        raw_image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        # raw_image = np.array(raw_image,dtype=np.uint8)
        image_name = image_path.split('/')[1]
        save_path = os.path.join(self.aug_path,image_path.split('/')[0]) + '/'
        if self.box and not self.withnosam:
            if os.path.exists(save_path+ image_name+'box_mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'box_mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")
                # mask_back = Image.open(save_path.replace("waterbird_aug_box", "waterbird_aug")+'mask_back.jpeg').convert("RGB")  # 消融实验，记得改回来
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM_box(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM_box(raw_image,save_path,'animal',0.1,0.1)
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")

        elif not self.withnosam:
            if os.path.exists(save_path+ image_name+'mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM(raw_image,save_path,'animal',0.1,0.1)
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
        
        if self.withnosam:
            mask_for = Image.open(os.path.join(self.root, image_path)).convert("RGB")
            mask_back = mask_for
        if self.raw_aug_num > 0 or self.for_aug_num > 0:
            # mask_back_list = [self.transform_(mask_back)]
            mask_back_list = [self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]
            mask_back_list_before_trans = []
            if self.raw_aug_num > 0:
                for i in range(self.raw_aug_num-1):
                    if self.withnosam:
                        mask_back_aug = self.nosam_aug(mask_back)
                    else:
                        mask_back_aug = self.raw_aug(mask_back)

                    mask_back_list.append(self.transform_(mask_back_aug))

            if self.for_aug_num > 0:
                mask_for_list = [self.transform_(mask_for)]
                for i in range(self.for_aug_num-1):
                    mask_for_list.append(self.transform_(self.for_aug(mask_for)))

            else:
                mask_for_list = self.transform_(mask_for)
            return {'image_path':image_path,
                    'raw':self.transform_(raw_image),
                    'for':mask_for_list,
                    'back':mask_back_list}
        else:
            if self.tpt_aug is not None:
                return {'image_path':image_path,
                        'raw':self.tpt_aug(raw_image),
                        'for':self.transform_(mask_for),
                        'back':self.transform_(mask_back)}
            else:
                return {'image_path':image_path,
                        'raw':self.transform_(raw_image),
                        'for':self.transform_(mask_for),
                        'back':[self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]}
        # return self.transform_(Image.open(x).convert("RGB"))

class crabspider(SubpopDataset):
    CHECKPOINT_FREQ = 1
    INPUT_SHAPE = (3, 224, 224,)

    def __init__(self, data_path, split, hparams=None, train_attr='yes', 
                        subsample_type=None, duplicates=None,transform=None,
                        box=False,shot=0,raw_aug=None,for_aug=None,tpt_aug=None,
                        withnosam=False,meta_part=None):
        self.use_anno = False
        root = os.path.join(data_path, "crab_spider")
        metadata = os.path.join(root, "metadata.csv")
        metadata = pd.read_csv(metadata)
        self.box = box
        self.aug_path = os.path.join(data_path, "crab_spider", "crab_spider_aug")
        if box:
            self.aug_path = os.path.join(data_path, "crab_spider", "crab_spider_aug_box")
        self.shot = shot
        # transform = transforms.Compose([
        #     transforms.Resize((int(224 * (256 / 224)), int(224 * (256 / 224)),)),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        self.data_type = "images"
        self.class_name = ['crab','spider']
        self.raw_aug_num = 0
        if raw_aug is not None:
            self.raw_aug = raw_aug['aug']
            self.raw_aug_num = raw_aug['num']
        self.for_aug_num = 0
        if for_aug is not None:
            self.for_aug = for_aug['aug']
            self.for_aug_num = for_aug['num']
            self.same = for_aug['same']

        self.tpt_aug = tpt_aug
        self.withnosam = withnosam
        self.nosam_aug = RandAugment(1, strong=True)
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates,meta_part=meta_part)

    def transform(self, image_path):
        raw_image = Image.open(os.path.join(self.root, image_path)).convert("RGB")
        # raw_image = np.array(raw_image,dtype=np.uint8)
        image_name = image_path.split('/')[1]
        save_path = os.path.join(self.aug_path,image_path.split('/')[0]) + '/'
        if self.box and not self.withnosam:
            if os.path.exists(save_path+ image_name+'box_mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'box_mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")
                # mask_back = Image.open(save_path.replace("waterbird_aug_box", "waterbird_aug")+'mask_back.jpeg').convert("RGB")  # 消融实验，记得改回来
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM_box(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM_box(raw_image,save_path,'animal',0.1,0.1)
                mask_for = Image.open(save_path+'box_mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'box_mask_back.jpeg').convert("RGB")

        elif not self.withnosam:
            if os.path.exists(save_path+ image_name+'mask_for.jpeg') \
                    and os.path.exists(save_path+ image_name+'mask_back.jpeg'):

                save_path = save_path + image_name
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
                # mask_for = np.array(Image.open(save_path+'mask_for.jpeg').convert("RGB"),dtype=np.uint8)
                # mask_back = np.array(Image.open(save_path+'mask_back.jpeg').convert("RGB"),dtype=np.uint8)
            else:
                os.makedirs(save_path,exist_ok=True)
                save_path = save_path + image_name
                # makefig_GSAM(raw_image,save_path,'bird',0.1,0.1)
                makefig_GSAM(raw_image,save_path,'animal',0.1,0.1)
                mask_for = Image.open(save_path+'mask_for.jpeg').convert("RGB")
                mask_back = Image.open(save_path+'mask_back.jpeg').convert("RGB")
        
        if self.withnosam:
            mask_for = Image.open(os.path.join(self.root, image_path)).convert("RGB")
            mask_back = mask_for
        if self.raw_aug_num > 0 or self.for_aug_num > 0:
            # mask_back_list = [self.transform_(mask_back)]
            mask_back_list = [self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]
            mask_back_list_before_trans = []
            if self.raw_aug_num > 0:
                for i in range(self.raw_aug_num-1):
                    if self.withnosam:
                        mask_back_aug = self.nosam_aug(mask_back)
                    else:
                        mask_back_aug = self.raw_aug(mask_back)
                    # mask_back_list_before_trans.append(mask_back_aug)
                    mask_back_list.append(self.transform_(mask_back_aug))

            if self.for_aug_num > 0:
                mask_for_list = [self.transform_(mask_for)]
                for i in range(self.for_aug_num-1):
                    mask_for_list.append(self.transform_(self.for_aug(mask_for)))

            else:
                mask_for_list = self.transform_(mask_for)
            return {'image_path':image_path,
                    'raw':self.transform_(raw_image),
                    'for':mask_for_list,
                    'back':mask_back_list}
        else:
            if self.tpt_aug is not None:
                return {'image_path':image_path,
                        'raw':self.tpt_aug(raw_image),
                        'for':self.transform_(mask_for),
                        'back':self.transform_(mask_back)}
            else:
                return {'image_path':image_path,
                        'raw':self.transform_(raw_image),
                        'for':self.transform_(mask_for),
                        'back':[self.transform_(mask_back),
                                self.transform_(transforms.ToPILImage()(torch.zeros((3,*mask_back.size))))]}
        # return self.transform_(Image.open(x).convert("RGB"))
