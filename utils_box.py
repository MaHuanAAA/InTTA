import numpy as np
import torch
import matplotlib.pyplot as plt
# import cv2
# import os
from torch.autograd import Variable
import torchvision

from PIL import Image

from lavis.processors import load_processor

from lavis.models import load_model_and_preprocess

from torchvision.transforms.functional import normalize, resize, to_pil_image,to_tensor

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def makefig_GSAM(raw_image,save_path,prompt,
                 BOX_THRESHOLD=0.25,TEXT_THRESHOLD=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    GROUNDING_DINO_CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./models/groundingdino_swint_ogc.pth"
    NMS_THRESHOLD = 0.8
    # BOX_THRESHOLD = 0.25
    # TEXT_THRESHOLD = 0.25
    # BOX_THRESHOLD = 0.1
    # TEXT_THRESHOLD = 0.1

    raw_image = np.array(raw_image,dtype=np.uint8)

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(raw_image)

    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    detections = grounding_dino_model.predict_with_classes(
        image=raw_image,
        classes=[prompt],
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    if len(detections.xyxy) == 0:
        detections = grounding_dino_model.predict_with_classes(
            image=raw_image,
            classes=[prompt],
            box_threshold=0.0,
            text_threshold=0.0
        )
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    # box_annotator = sv.BoxAnnotator()
    # labels = [
    #     f"{[prompt][class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _ 
    #     in detections]
    # annotated_frame = box_annotator.annotate(scene=raw_image.copy(), detections=detections, labels=labels)
    # cv2.imwrite((save_path+'boxes.jpeg'), annotated_frame)

    # point_coords = np.array([[anno[0],anno[1]],[anno[0],anno[3]-1],
    #                          [anno[2]-1,anno[1]],
    #                          [anno[2]-1,anno[3]-1],
    #                          [(anno[0]+anno[2]-1)//2,(anno[1]+anno[3]-1)//2]]) 
    # point_labels = np.array([0,0,0,0,1]) 

    # plt.figure(figsize=(10,10))
    # plt.imshow(raw_image)
    # # show_points(point_coords, point_labels, plt.gca())
    # for i in range(len(detections.xyxy)):
    #     show_box(np.array(detections.xyxy[i]),plt.gca())
    # plt.axis('on')
    # plt.show()
    # plt.savefig("./point.png")
    masks_list = []
    for box in detections.xyxy:
        masks_, iou_predictions_, _ = predictor.predict( box=box,
                                                    multimask_output=True)
        mask_to_choose = np.argmax(iou_predictions_)
        masks_ = np.array(masks_,dtype=np.uint8)
        masks_list.append(masks_[mask_to_choose,:,:])
    
    masks = masks_list[0]
    if len(masks_list)>1:
        for items in masks_list[1:]:
            masks = np.logical_or(masks,items)
    masks = np.array(masks,dtype=np.uint8)
    # for i, (mask, score) in enumerate(zip(masks, iou_predictions)):
    #     plt.figure(figsize=(10,10))
    #     plt.imshow(raw_image)
    #     show_mask(mask, plt.gca())
    #     show_points(point_coords, point_labels, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()
    #     plt.savefig("./fig{}_{:.3f}.png".format(str(i),score))


    mask_image_for = Image.fromarray(raw_image*np.expand_dims(masks[:,:], -1))
    mask_image_for.save(save_path+'mask_for.jpeg')


    mask_image_back = Image.fromarray(raw_image*np.expand_dims(1-masks[:,:], -1))
    mask_image_back.save(save_path+'mask_back.jpeg')

def makefig_GSAM_box(raw_image,save_path,prompt,
                 BOX_THRESHOLD=0.25,TEXT_THRESHOLD=0.25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "./models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    GROUNDING_DINO_CONFIG_PATH = "./groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./models/groundingdino_swint_ogc.pth"
    NMS_THRESHOLD = 0.8
    # BOX_THRESHOLD = 0.25
    # TEXT_THRESHOLD = 0.25
    # BOX_THRESHOLD = 0.1
    # TEXT_THRESHOLD = 0.1

    raw_image = np.array(raw_image,dtype=np.uint8)

    sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(raw_image)

    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, 
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    detections = grounding_dino_model.predict_with_classes(
        image=raw_image,
        classes=[prompt],
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    if len(detections.xyxy) == 0:
        detections = grounding_dino_model.predict_with_classes(
            image=raw_image,
            classes=[prompt],
            box_threshold=0.0,
            text_threshold=0.0
        )
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    masks = np.zeros(raw_image.shape[:2])
    for box in detections.xyxy:
        temp = np.zeros(masks.shape)
        # temp[int(box[0]):int(box[2]),int(box[1]):int(box[3])] = 1
        temp[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = 1
        masks = np.logical_or(masks,temp)
    masks = np.array(masks,dtype=np.uint8)


    mask_image_for = Image.fromarray(raw_image*np.expand_dims(masks[:,:], -1))
    mask_image_for.save(save_path+'box_mask_for.jpeg')


    mask_image_back = Image.fromarray(raw_image*np.expand_dims(1-masks[:,:], -1))
    mask_image_back.save(save_path+'box_mask_back.jpeg')


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def zero_shot_tune_def_ent(raw_image,cls_names, model, vis_processors, text_processors,
                   show_case=False,lr=5e-2,train_epoch=5,embed_l=2,dataset_name='blip',
                   embed=None,caption = "a photo of ",label=False,same=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_l = 3
    criterion = torch.nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    kl_loss_tarlog = nn.KLDivLoss(reduction="batchmean",log_target=True)

    caption_list = [text_processors["eval"](caption+i) for i in cls_names]
    # sample = {"image": image, "text_input": caption_list}
    raw_sample = {"image": raw_image['raw'].to(device), "text_input": caption_list}
    for_sample = {"image": raw_image['for'].to(device), "text_input": caption_list}
    back_sample = {"image": raw_image['back'].to(device), "text_input": caption_list}
    if embed == None:
        if 'clip' in dataset_name:
            text = model.tokenizer(caption_list).to(model.device)
            n_ctx = torch.sum(text!=0)/2
            embedding_output = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
            # print(n_ctx)
            n_ctx = int(n_ctx.item())
            init_embed = embedding_output
            param_embed = torch.zeros(embed_l,init_embed.shape[-1]).cuda()
            now_embed = init_embed + model.positional_embedding
            param_embed[:,:] = now_embed[0,1:1+embed_l,:]
            
            param_embed = Variable(param_embed, requires_grad=True)
            # print(param_embed)
            optimizer = torch.optim.AdamW([param_embed], lr)
        else:
            text = model.tokenizer(caption_list, return_tensors="pt", padding=True).to(
                        model.device
                    )
            _, embedding_output= model.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            init_embed = embedding_output
            param_embed = torch.zeros(embed_l,init_embed.shape[-1]).cuda()
            now_embed = init_embed
            param_embed[:,:] = now_embed[0,1:1+embed_l,:]
            param_embed = Variable(param_embed, requires_grad=True)
            # print(param_embed)
            optimizer = torch.optim.AdamW([param_embed], lr)
    else:
        class_embed = []
        if isinstance(embed,list):
            [embed,class_embed] = embed
        if 'clip' in dataset_name:
            text = model.tokenizer(caption_list).to(model.device)
            n_ctx = torch.sum(text!=0)/2
            embedding_output = model.token_embedding(text)  
            # print(n_ctx)
            n_ctx = int(n_ctx.item())
            init_embed = embedding_output
            now_embed = init_embed + model.positional_embedding
            for i in range(len(cls_names)):
                now_embed[i,1:embed_l+1,:] = embed
                if class_embed != []:
                    now_embed[i,1+embed_l:n_ctx-1,:] = class_embed[i]
        else:
            text = model.tokenizer(caption_list, return_tensors="pt", padding=True).to(
                        model.device
                    )
            _, embedding_output= model.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            init_embed = embedding_output
            now_embed = init_embed
            for i in range(len(cls_names)):
                now_embed[i,1:embed_l+1,:] = embed
                if class_embed != []:
                    now_embed[i,embed_l+1:-1,:] = class_embed[i]
        
        param_embed = torch.zeros(embed_l,init_embed.shape[-1]).cuda()
        param_embed[:,:] = now_embed[0,1:1+embed_l,:]
        param_embed = Variable(param_embed, requires_grad=True)
        optimizer = torch.optim.AdamW([param_embed], lr)
    ep = np.linspace(0, 1, train_epoch+1)
    for n in range(train_epoch):
        optimizer.zero_grad()
        # now_embed = init_embed.clone()
        loss = 0
        for i in range(len(cls_names)):
            now_embed[i,1:embed_l+1,:] = param_embed
        # raw_features = model.extract_features(raw_sample, mode="image").image_embeds_proj.mean(dim=1)
        # text_features = model.extract_features(raw_sample, mode="text",embed=now_embed).text_embeds_proj.mean(dim=1)
        
        # probs = (raw_features @ text_features.t()) / model.temp
        # loss += criterion(probs, torch.tensor([1]).cuda())

        raw_features = model.extract_features(raw_sample, mode="image").image_embeds_proj.mean(dim=1)
        for_features = model.extract_features(for_sample, mode="image").image_embeds_proj.mean(dim=1)
        back_features = model.extract_features(back_sample, mode="image").image_embeds_proj.mean(dim=1)
        
        text_features = model.extract_features(raw_sample, mode="text",embed=now_embed).text_embeds_proj.mean(dim=1)
        text_features_base = model.extract_features(raw_sample, mode="text",embed=None).text_embeds_proj.mean(dim=1)
        probs = (back_features @ text_features.t()) / model.temp
        # probs = (back_features @ text_features.t()) /0.1
        probs = F.log_softmax(probs, dim=1)
        # loss += criterion(probs, torch.ones(probs.shape).cuda()/len(cls_names))

        loss += kl_loss(probs, torch.ones(probs.shape).cuda()/len(cls_names))
        if same:
            probs_for = (for_features @ text_features.t()) / model.temp
            probs_for = (torch.nn.functional.softmax(probs_for, dim=1)+1e-8).log()
            # assert torch.sum(torch.isnan(probs_for_back)) == 0
            # probs = (torch.nn.functional.softmax((probs.detach())[0::2,:],dim=1)+1e-8).log()
            probs_for_base = (for_features @ text_features_base.t()) / model.temp
            probs_for_base = (torch.nn.functional.softmax(probs_for_base,dim=1)+1e-8).log()
            loss += kl_loss_tarlog(probs_for, probs_for_base.detach())

            # probs = (for_features @ text_features.t()) / model.temp
            # probs_for_back = probs[1::2,:]
            # probs = probs[0::2,:]
            # loss += criterion(probs_for_back, torch.nn.functional.softmax(probs.detach(),dim=1))
            # # loss += criterion(probs_for_back, torch.nn.functional.softmax(probs,dim=1))
            # loss += criterion(probs, torch.nn.functional.softmax(probs,dim=1))
        else:
            probs = (for_features @ text_features.t()) / model.temp
            loss += criterion(probs, torch.nn.functional.softmax(probs,dim=1))
            # loss += avg_entropy(probs)
        # loss += - (probs.softmax(1) * probs.log_softmax(1)).sum(1).mean()
        # print(- (probs.softmax(1) * probs.log_softmax(1)).sum(1).mean(),' ',
        #       criterion(probs, torch.nn.functional.softmax(probs,dim=1)))

        # probs = (for_features @ text_features.t())/0.1
        # loss += criterion(probs, torch.nn.functional.softmax(probs,dim=1))
        # loss += criterion(probs, torch.nn.functional.softmax(probs.detach_(),dim=1))
        if label:
            if len(probs.detach_()) > 1 :
                probs = torch.mean(probs.detach_(),dim=0,keepdim=True)
            _,y_for = torch.max(probs.detach_(), 1)
            probs = (raw_features @ text_features.t()) / model.temp
            loss += criterion(probs, y_for)

        # probs = (raw_features @ text_features.t()) / model.temp
        # loss += criterion(probs, torch.nn.functional.softmax(probs,dim=1))
        
        # probs = (raw_features @ text_features.t()) / model.temp
        # loss += criterion(probs, torch.nn.functional.softmax(probs.detach_(),dim=1))
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f"train_{n}: \t {loss.item():.3}")

    # probs_show = torch.nn.Softmax(dim=1)(sims)[0,:]
    # for cls_nm, prob in zip(cls_names, probs_show):
    #     print(f"{cls_nm}: \t {prob.item():.3%}")
    if embed == None:
        return param_embed.detach_().clone()
    else:
        return [param_embed.detach_().clone(),class_embed]


@torch.no_grad()
def zero_shot_tune_test_def(image,cls_names, model, vis_processors, text_processors,
                   show_case=False,embed=None,embed_l=2,
                   dataset_name='blip',caption = "a photo of "):

    embed_l = 3
    # cls_names = ["waterbird", "landbird"]
    caption_list = [text_processors["eval"](caption+i) for i in cls_names]
    # caption_list = ["This is not a picture of waterbird","This is a picture of waterbird"]
    sample = {"image": image, "text_input": caption_list}
    class_embed = []
    if isinstance(embed,list):
        [embed,class_embed] = embed
    if 'clip' in dataset_name:
        text = model.tokenizer(caption_list).to(model.device)
        # embedding_output = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # embedding_output = embedding_output + model.positional_embedding
        n_ctx = torch.sum(text!=0)/2
        embedding_output = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # print(n_ctx)
        n_ctx = int(n_ctx.item())
        init_embed = embedding_output
        now_embed = init_embed + model.positional_embedding
        for i in range(len(cls_names)):
            now_embed[i,1:embed_l+1,:] = embed
            if class_embed != []:
                now_embed[i,1+embed_l:n_ctx-1,:] = class_embed[i]

    else:
        text = model.tokenizer(caption_list, return_tensors="pt", padding=True).to(
                    model.device
                )
        _, embedding_output= model.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        init_embed = embedding_output
        now_embed = init_embed
        for i in range(len(cls_names)):
            now_embed[i,1:embed_l+1,:] = embed
            if class_embed != []:
                now_embed[i,embed_l+1:-1,:] = class_embed[i]


    # image_features = model.extract_features(sample, mode="image").image_embeds_proj[:, 0]
    # text_features = model.extract_features(sample, mode="text",embed=init_embed).text_embeds_proj[:, 0]
    image_features = model.extract_features(sample, mode="image").image_embeds_proj.mean(dim=1)
    text_features = model.extract_features(sample, mode="text",embed=now_embed).text_embeds_proj.mean(dim=1)

    sims = (image_features @ text_features.t()) / model.temp
    # sims = (image_features @ text_features.t()) 
    probs = sims
    if show_case:
        probs_show = torch.nn.Softmax(dim=1)(sims)[0,:]

        for cls_nm, prob in zip(cls_names, probs_show):
            print(f"{cls_nm}: \t {prob.item():.3%}")
    return probs


    



