import torch
import numpy as np
from utils_box import *
from Data.datasets import *
from lavis.models import load_model_and_preprocess
import pandas as pd
from randaugment import RandAugment
import argparse

if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def get_arg():
        parser = argparse.ArgumentParser(description = 'test')
        parser.add_argument('--dataset_name', default='Waterbirds_clipB32')
        parser.add_argument('--seed', default=0,type=int)
        parser.add_argument('--root_path', default='root',
                            help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
        parser.add_argument('--save_path', default='sve',
                            help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')
        parser.add_argument('--LR', type=float, default=5e-4)
        parser.add_argument('--EPOCH', type=int, default=20)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--SHOT', type=int, default=0)
        parser.add_argument('--RAW_AUG', default=False, type=str2bool)
        parser.add_argument('--RAW_AUG_NUM', type=int, default=16)
        parser.add_argument('--FOR_AUG', default=False, type=str2bool)
        parser.add_argument('--FOR_AUG_NUM', type=int, default=16)
        parser.add_argument('--NOSAM', default=False, type=str2bool)
        parser.add_argument('--LABEL', default=False, type=str2bool)
        parser.add_argument('--SAME_DISTRU', default=False, type=str2bool)
        parser.add_argument('--USE_CLASS', default=False, type=str2bool)
        
        config = parser.parse_args()
        return config


    args = get_arg()
    root_path = args.root_path
    dataset_name = args.dataset_name

    seed = args.seed
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
 
    LR = args.LR
    EPOCH = args.EPOCH
    SHOT = args.SHOT
    EMBED_LEN = 3
    print(args.RAW_AUG)
    RAW_AUG = args.RAW_AUG
    RAW_AUG_NUM = args.RAW_AUG_NUM
    FOR_AUG = args.FOR_AUG
    FOR_AUG_NUM = args.FOR_AUG_NUM
    SAME_DISTRU = args.SAME_DISTRU
    LABEL = args.LABEL
    USE_CLASS = args.USE_CLASS


    if not RAW_AUG:
        RAW_AUG_NUM = 0
    if not FOR_AUG:
        FOR_AUG_NUM = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # eva_vit.py blip2_qformer.py
    if 'clipL14' in dataset_name:
        model, vis_processors, text_processors = load_model_and_preprocess("clip_feature_extractor", 
                                                                        model_type="ViT-L-14", 
                                                                        is_eval=True, 
                                                                        device=device)
    elif 'clipB32' in dataset_name:
        model, vis_processors, text_processors = load_model_and_preprocess("clip_feature_extractor", 
                                                                        model_type="ViT-B-32", 
                                                                        is_eval=True, 
                                                                        device=device)
    else:
        model, vis_processors, text_processors = load_model_and_preprocess("blip2_feature_extractor",
                                                                        "pretrain",
                                                                        device=device,
                                                                        is_eval=True)
    if SHOT > 0:
        dataset_train = eval(dataset_name.split('_')[0])(data_path=root_path, 
                                                split='tr',
                                                transform=vis_processors["eval"],
                                                box='box' in dataset_name,
                                                shot=SHOT)
        print("dataset train :",len(dataset_train))
        cls_name = dataset_train.class_name
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, 
            num_workers=0, pin_memory=True,
        )

        embed = few_shot_tune_def(cls_name, model, vis_processors, text_processors,
                   show_case=False,lr=LR,train_epoch=EPOCH,data_loader=data_loader_train,
                   use_class=USE_CLASS,embed_l=EMBED_LEN,dataset_name=dataset_name,
                   caption=caption_dict[dataset_name.split('_')[0]])

    raw_aug = None
    for_aug = None
    if RAW_AUG:
        raw_aug = {'aug':RandAugment(2),
                    'num':RAW_AUG_NUM,}
    if FOR_AUG:
        for_aug = {'aug':RandAugment(2),
                    'num':FOR_AUG_NUM,
                    'same':SAME_DISTRU,}
    dataset = eval(dataset_name.split('_')[0])(data_path=root_path, 
                                                split='te',
                                                transform=vis_processors["eval"],
                                                box='box' in dataset_name,
                                                raw_aug=raw_aug,
                                                for_aug=for_aug, withnosam=args.NOSAM)
    cls_name = dataset.class_name

    data_loader = torch.utils.data.DataLoader(
        # dataset, batch_size=32, 
        dataset, batch_size=args.batch_size, 
        num_workers=0, pin_memory=False, shuffle=False
    )
    result = []
    result_before = []
    result_for = []
    result_back = []
    image_mean = []
    image_mean_for = []
    image_mean_back = []
    image_path = []
    atts = []
    gs = []
    ys = []
    embed_list=[]
    # result_t = []
    # with torch.no_grad():
    for i, (img_id,raw_image, y, a) in enumerate(data_loader):
        print("batch:",i)
        print(raw_image['image_path'])
        if isinstance(raw_image['back'],list):
            raw_image['back'] = torch.concat(raw_image['back'],axis=0)
        if isinstance(raw_image['for'],list):
            raw_image['for'] = torch.concat(raw_image['for'],axis=0)


        if SHOT == 0:

            embed = zero_shot_tune_def_ent(raw_image,cls_name, model, vis_processors, text_processors,
                                    show_case=False,lr=LR,train_epoch=EPOCH,
                                    embed_l=EMBED_LEN,dataset_name=dataset_name,
                                    caption=caption_dict[dataset_name.split('_')[0]],
                                    label=LABEL,same=SAME_DISTRU)

        
        prob_raw = zero_shot_tune_test_def(raw_image['raw'].to(device),cls_name,
                                    model, vis_processors, text_processors,embed=embed,
                                    embed_l=EMBED_LEN,dataset_name=dataset_name,
                                    caption=caption_dict[dataset_name.split('_')[0]])
        print('raw',torch.nn.functional.softmax(prob_raw, dim=1))
        prob_for = zero_shot_tune_test_def(raw_image['for'].to(device),cls_name,
                                    model, vis_processors, text_processors,embed=embed,
                                    embed_l=EMBED_LEN,dataset_name=dataset_name,
                                    caption=caption_dict[dataset_name.split('_')[0]])
        print('for', torch.nn.functional.softmax(prob_for, dim=1))
        prob_back = zero_shot_tune_test_def(raw_image['back'].to(device),cls_name,
                                    model, vis_processors, text_processors,embed=embed,
                                    embed_l=EMBED_LEN,dataset_name=dataset_name,
                                    caption=caption_dict[dataset_name.split('_')[0]])
        print('back', torch.nn.functional.softmax(prob_back, dim=1))

        result.append(prob_raw.cpu())
        result_for.append(prob_for.cpu())
        result_back.append(prob_back.cpu())
        embed_list.append(embed.unsqueeze(0).cpu())
        image_path.append(raw_image['image_path'])

        ys.append(y)
        atts.append(a)
        gs.append([f'y={yi},a={gi}' for c, (yi, gi) in enumerate(zip(y, a))])

    np.save(args.save_path + '/npys/scaletemp_{}_kl_{}epoch_{}_{}_finetune_rand_shot{}_r{}_f{}_{}_bs{}_seed{}_{}.npy'.format(
                                                                 'label' if LABEL else '',
                                                                 EPOCH,
                                                                 LR,
                                                                 dataset_name,
                                                                 SHOT,
                                                                 RAW_AUG_NUM if RAW_AUG else '',
                                                                 FOR_AUG_NUM if FOR_AUG else '',
                                                                 'sameklnodetach' if SAME_DISTRU else '',str(args.batch_size),str(args.seed),'nosam' if args.NOSAM else ''),{
        'raw':np.concatenate(result,axis=0),
        'for':np.concatenate(result_for,axis=0),
        'back':np.concatenate(result_back,axis=0),
        'class':np.concatenate(ys, axis=0), 
        'atts':np.concatenate(atts, axis=0),
        'gs': np.concatenate(gs),
        'image_path': np.concatenate(image_path),
        'embed':np.concatenate(embed_list,axis=0)
    })
        


