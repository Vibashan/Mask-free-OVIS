'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import argparse
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import numpy as np
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import sys
sys.path.append("ALBEF/")
import torch.backends.cudnn as cudnn

from functools import partial
from ALBEF.models.vit import VisionTransformer
from ALBEF.models.xbert import BertConfig, BertModel
from ALBEF.models.tokenization_bert import BertTokenizer

from ALBEF import utils
from ALBEF.dataset import create_dataset, create_sampler, create_loader
import pickle
import cv2
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

import copy
import math
import torch.optim as optim
from torch.autograd import Variable

class VL_Transformer_ITM(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 config_bert='',
                 img_size=384
                 ):
        super().__init__()

        bert_config = BertConfig.from_json_file(config_bert)
        self.visual_encoder = VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        self.itm_head = nn.Linear(768, 2)

    def forward(self, image, text):
        image_embeds = self.visual_encoder(image)

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids,
                                   attention_mask=text.attention_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask=image_atts,
                                   return_dict=True,
                                   )

        vl_embeddings = output.last_hidden_state[:, 0, :]
        vl_output = self.itm_head(vl_embeddings)
        return vl_output

class WSS_Net(nn.Module):
    def __init__(self,input_dim):
        super(WSS_Net, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1 ) 
        self.bn2 =  nn.BatchNorm2d(128) 
        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
    
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x
    
def act_map_2_pseduo_gt(seg_map):
    pseduo_gt = np.ones((seg_map.shape[0], seg_map.shape[1]))
    pseduo_gt = np.uint8(255 * pseduo_gt)
    h, w  = seg_map.shape
    shift = 5
    cv2.rectangle(pseduo_gt, (shift,shift), (w-shift,h-shift), (0, 0, 0))

    _, counts = np.unique(pseduo_gt, return_counts=True)
    bg_count = counts[0]
    fg_count = bg_count*1
    seg_idx = np.where(seg_map > 0)

    if len(seg_idx[0]) > 0 :
        if len(seg_idx[0]) < fg_count:
            total_pts = len(seg_idx[0])-1
        else:
            total_pts = fg_count
        
        sample_pts = np.random.randint(len(seg_idx[0])-1, size=total_pts)
        sample_x, sample_y = seg_idx[0][sample_pts], seg_idx[1][sample_pts]
        pseduo_gt[sample_x,sample_y] = 8
        cv2.circle(pseduo_gt,(w//2, h//2), 1, (8,8,8), -1)
    else:
        cv2.circle(pseduo_gt,(w//2, h//2), 5, (8,8,8), -1)
    return pseduo_gt

def wss_pipeline(cropped_img, crop_act_map, file_name):
    
    nChannel = 128
    maxIter = 400
    minLabels = 3
    
    im = cropped_img.cpu().numpy()
    im = im.transpose(1,2,0)
    im = (im - im.min()) / (im.max() - im.min())
    im = np.uint8(255 * im)
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    data = Variable(cropped_img.unsqueeze(0)).cuda()

    pseduo_gt_mask = act_map_2_pseduo_gt(np.uint8(crop_act_map*255))
    mask = pseduo_gt_mask.reshape(-1)
    mask_inds = np.unique(mask) 
    mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
    inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )  ##### Background idx
    inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )  ##### Foreground idx
    target_scr = torch.from_numpy( mask.astype(np.int) )
    inds_sim = inds_sim.cuda()
    inds_scr = inds_scr.cuda()
    target_scr = target_scr.cuda()
    target_scr = Variable(target_scr)
    minLabels = len(mask_inds)

    # train
    model = WSS_Net(data.size(1))
    model.cuda()
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn_scr = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0.9)
    for batch_idx in range(maxIter):
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        
        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))

        loss =  loss_fn(output[ inds_sim ], target[ inds_sim ]) + loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) #+ (lhpy + lhpz)
        loss.backward()
        optimizer.step()

        if nLabels <= minLabels:
            break

    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
    _, target = torch.max( output, 1 )
    
    im_target = target.data.cpu().numpy()
    im_instance_mask = im_target.reshape( im.shape[0], im.shape[1] ).astype( np.uint8 )
    im_instance_mask[im_instance_mask !=8] = 0

    return im_instance_mask

def net_normalized_act_map(total_act_map_obj):
    sum_act_map_obj = 0
    for i in range(len(total_act_map_obj)):
        req_act_map = total_act_map_obj[i]
        req_act_map = (req_act_map - req_act_map.min()) / (req_act_map.max() - req_act_map.min())
        sum_act_map_obj = req_act_map+sum_act_map_obj
    return sum_act_map_obj

def impaint_function(act_map_obj, img):
    net_img_mean = img.mean()
    
    act_map_obj = act_map_obj.detach().clone().squeeze().cuda()
    act_map_obj = (act_map_obj - act_map_obj.min()) / (act_map_obj.max() - act_map_obj.min())
    act_map_obj[act_map_obj < 0.5] = 0.0  
    act_map_obj[act_map_obj > 0.5] = 1.0
    
    mask = act_map_obj
    inv_mask = mask.detach().clone()
    inv_mask[mask==0] = 1
    inv_mask[mask==1] = 0
    
    mask_img = mask*img
    impaint = mask_img.detach().clone()
    impaint[impaint!=0] = net_img_mean
    
    inv_mask_img = inv_mask*img
    net_img = inv_mask_img+impaint
    
    return net_img

def seg_2_poly(instance_mask):
    instance_mask = np.uint8(instance_mask)
    instance_mask = cv2.GaussianBlur(instance_mask,(5,5),0)
    contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        if contour.size >= 6:
            segmentation.append(contour.ravel().tolist())
    return segmentation

def vis_det_act(image_, image_relevance, bbox, text, filename, output_dir, bbox_prop = None, instance_mask = None):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    image_ = image_.unsqueeze(0)
    image = F.interpolate(image_, size=(image_relevance.shape[-2],image_relevance.shape[-1]))
    image = image.squeeze(0)
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    #vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * image)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    fig, ax = plt.subplots()
    ax.imshow(vis)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=3, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    if bbox_prop is not None:
        for bbp in bbox_prop:
            rect = patches.Rectangle((bbp[0], bbp[1]), bbp[2]-bbp[0], bbp[3]-bbp[1], linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
    plt.text(bbox[0]-5, bbox[1]-5, text, color='white', fontsize=15)
    plt.axis('off')
    
    if len(instance_mask) > 0: 
        for seg in instance_mask:
            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
            polygons = patches.Polygon(poly)
            p = PatchCollection([polygons], facecolor='r', linewidths=0, alpha=0.6)
            ax.add_collection(p)
            p = PatchCollection([polygons], facecolor='none', edgecolors='b', linewidths=0.5)
            ax.add_collection(p)    
    if not os.path.isdir(os.path.join(output_dir+'vis')):
        os.makedirs(os.path.join(output_dir+'vis'))
    plt.savefig(os.path.join(output_dir+'vis', filename.split('.')[0]+'_{}.png'.format(text.replace('/', '_'))))
    print("Saved Image with Box-level and Pixel-level Annotations in ", os.path.join(output_dir+'vis', filename.split('.')[0]+'_{}.png'.format(text.replace('/', '_'))))

    
def get_activation_map(output, model, image, text_input_mask, block_num, map_size, batch_index):
    loss = output[1].sum()
    image = image.unsqueeze(0)
    text_input_mask = text_input_mask.unsqueeze(0)

    model.zero_grad()
    loss.backward(retain_graph=True)

    with torch.no_grad():
        mask = text_input_mask.view(text_input_mask.size(0),1,-1,1,1)

        grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
        cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()
        cams = cams[batch_index, :, :, 1:].reshape(image.size(0), 12, -1, map_size, map_size)
        cams = cams * mask
        grads = grads[batch_index, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, map_size, map_size) * mask

        gradcam = cams * grads
        gradcam = gradcam.mean(1)
    return gradcam[0, :, :, :].cpu().detach()

def generate_pseudo_bbox(model, tokenizer, data_loader, object_name_dict, args, block_num, map_size, device):
    num_image_without_proposals = 0
    num_image = 0
 
    metric_logger = utils.MetricLogger(delimiter="  ")
    print_freq = 50

    tokenized_dict = {}
    for (k,v_list) in object_name_dict.items():
        tokenized_v_list = []
        for v in v_list:
            value_tmp = tokenizer._tokenize(v)
            value = ' '.join(value_tmp)
            tokenized_v_list.append(value)
        tokenized_dict[k] = tokenized_v_list


    for batch_i, (images, text, proposal_paths) in enumerate(metric_logger.log_every(data_loader, print_freq, '')):
        
        original_img = copy.deepcopy(images)
        
        objects_dict = {} # key is the proposal_path
        objects = []
        for (i, proposal_path) in enumerate(proposal_paths):
            wl = tokenizer._tokenize(text[i])
            tokenizeded_text = ' '.join(wl)
            tokenizeded_text = ' ' + tokenizeded_text + ' '
            objects_for_one = []
            # for every value token, see if there is an exact match
            for (k, v_list) in tokenized_dict.items():
                for v in v_list:
                    left_index = tokenizeded_text.find(' '+v+' ')
                    if left_index != -1:
                        space_count = tokenizeded_text[:(left_index+1)].count(' ')
                        objects_for_one.append((k,v, space_count, space_count+len(v.strip().split(' '))))
            objects.append(objects_for_one)
            
        ########################################## Iterative Masking ##################################################
        total_iter = 3
        mask_dict = {}
        bbox_dict = {}
        box_cnt_thresh = 1
        for cnt in range(total_iter):
            image = images
            image = image.to(device, non_blocking=True) 
            text_input = tokenizer(text, padding='longest', max_length=30, return_tensors="pt").to(device)
            output = model(image, text_input)

            impaint_img = []
            for i, img in enumerate(image):
                filename = proposal_paths[i].split('/')[-1]

                im_h, im_w = img.shape[1], img.shape[2]
                act_map = get_activation_map(output[i], model, img, text_input['attention_mask'][i], block_num, map_size, i)

                list_bbox_act_map_obj = []
                for (original_obj_name, replaced_obj_name, obj_i_left, obj_i_right) in objects[i]: 
                    
                    file_object = filename.split('.')[0]+"_"+original_obj_name
                    act_map_obj = act_map[obj_i_left]
                    
                    if obj_i_right - obj_i_left > 1:
                        for obj_i in range(obj_i_left+1, obj_i_right):
                            act_map_obj += act_map[obj_i]
                    mask_act_map_obj = F.interpolate(act_map_obj.unsqueeze(0).unsqueeze(0), size=(im_h, im_w), mode='bilinear').detach().clone()
                    bbox_act_map_obj = F.interpolate(act_map_obj.unsqueeze(0).unsqueeze(0), size=(im_h, im_w)).detach().clone()
                    
                    list_bbox_act_map_obj.append(bbox_act_map_obj)
                    if file_object not in mask_dict:
                        
                        mask_act_map_obj = (mask_act_map_obj - mask_act_map_obj.min()) / (mask_act_map_obj.max() - mask_act_map_obj.min())
                        
                        mask_act_map_obj_numpy = np.uint8(mask_act_map_obj.numpy().squeeze()*255)
                        mask_act_map_obj_numpy = cv2.GaussianBlur(mask_act_map_obj_numpy,(5,5),0)
                        _, mask_act_map_obj_numpy = cv2.threshold(mask_act_map_obj_numpy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        mask_act_map_obj_numpy[mask_act_map_obj_numpy==255] = 1
                        mask_act_map_obj = torch.from_numpy(mask_act_map_obj_numpy).unsqueeze(0).unsqueeze(0).float()
                        mask_dict[file_object] = [mask_act_map_obj]
                        bbox_dict[file_object] = [(bbox_act_map_obj - bbox_act_map_obj.min()) / (bbox_act_map_obj.max() - bbox_act_map_obj.min())]
                    elif file_object in mask_dict:
                        mask_act_map_obj = (mask_act_map_obj - mask_act_map_obj.min()) / (mask_act_map_obj.max() - mask_act_map_obj.min())
                        
                        mask_act_map_obj_numpy = np.uint8(mask_act_map_obj.numpy().squeeze()*255)
                        mask_act_map_obj_numpy = cv2.GaussianBlur(mask_act_map_obj_numpy,(5,5),0)
                        _, mask_act_map_obj_numpy = cv2.threshold(mask_act_map_obj_numpy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        mask_act_map_obj_numpy[mask_act_map_obj_numpy==255] = 1
                        mask_act_map_obj = torch.from_numpy(mask_act_map_obj_numpy).unsqueeze(0).unsqueeze(0).float()
                        mask_dict[file_object].append(mask_act_map_obj)
                        if cnt < box_cnt_thresh:
                            bbox_dict[file_object].append((bbox_act_map_obj - bbox_act_map_obj.min()) / (bbox_act_map_obj.max() - bbox_act_map_obj.min()))
                
                net_act_map_obj = net_normalized_act_map(list_bbox_act_map_obj)
                if len(objects[i]) == 0:
                    impaint_img.append(img)
                else:
                    impaint_img.append(impaint_function(net_act_map_obj, img))
            images = torch.stack(impaint_img)

        ########################################## Proposal Read ##################################################
        for i, img in enumerate(image):
            filename = proposal_paths[i].split('/')[-1]
            nearest_folder = proposal_paths[i].split('/')[-2]
            _, file_extension = os.path.splitext(proposal_paths[i])
            if file_extension == '':
                proposal_addr = proposal_paths[i]+'.pkl'
                info_addr = proposal_paths[i]+'_info.pkl'
            else:
                proposal_addr = proposal_paths[i].replace(file_extension,'.pkl')
                info_addr = proposal_paths[i].replace(file_extension,'_info.pkl')
            if not os.path.exists(proposal_addr):
                num_image_without_proposals += 1
                continue
            initial_proposals = pickle.load(open(proposal_addr, 'rb'))
            initial_information = pickle.load(open(info_addr, 'rb'))
            im_h, im_w = initial_information['ori_shape'][:2]
            proposals = []
            for p in initial_proposals:
                if p.size != 0:
                    proposals.extend(p)
            if len(proposals) == 0:
                num_image_without_proposals += 1
                continue
            proposals = np.stack(proposals, axis=0)
            prop_boxes = proposals[:,0:4]
            
            ########################################## Best Proposal Selection ##################################################

            num_image += 1
            print("Processed " +str(num_image) + " images")
            object_pseudo_list_per_image = []
            for (original_obj_name, replaced_obj_name, obj_i_left, obj_i_right) in objects[i]: 
                
                file_object = filename.split('.')[0]+"_"+original_obj_name
                act_map_obj = sum(bbox_dict[file_object])
                act_map_obj = F.interpolate(act_map_obj, size=(im_h, im_w)).cpu().numpy()
                act_map_obj = act_map_obj.squeeze()
                
                instance_act_map_obj = sum(mask_dict[file_object])
                instance_act_map_obj = F.interpolate(instance_act_map_obj, size=(im_h, im_w), mode='bilinear').cpu().numpy()
                instance_act_map_obj = instance_act_map_obj.squeeze()
                instance_act_map_obj[instance_act_map_obj > 0] = 1
                
                score_max = -1
                best_proposal = [0, 0, 0, 0]
                
                act_map_obj = (act_map_obj - act_map_obj.min()) / (act_map_obj.max() - act_map_obj.min())
                act_map_obj = np.uint8(act_map_obj*255)
                act_map_obj = cv2.GaussianBlur(act_map_obj,(5,5),0)
                _, act_map_obj = cv2.threshold(act_map_obj,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                act_map_obj[act_map_obj==255] = 1

                for bi, bb in enumerate(prop_boxes):
                    bb_tmp = np.copy(bb)  
                    area = float(bb_tmp[2] - bb_tmp[0]) * float(bb_tmp[3] - bb_tmp[1])
                    if bb_tmp[0] < 0 or bb_tmp[1] < 0 or bb_tmp[2] > act_map_obj.shape[1] or bb_tmp[3] > act_map_obj.shape[0]:
                        continue
                    det_score = act_map_obj[int(bb_tmp[1]):int(bb_tmp[3]), int(bb_tmp[0]):int(bb_tmp[2])]
                    if len(det_score) == 0 or area == 0:
                        continue
                    det_score = det_score.sum() / math.sqrt(area)
                    if det_score > score_max:
                        score_max = det_score
                        best_proposal = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
                        
                ########################################## Mask Generation ##################################################

                crop_act_map = instance_act_map_obj[best_proposal[1]:best_proposal[3],best_proposal[0]:best_proposal[2]]
                resize_transform = T.Resize((im_h, im_w))
                resized_img = resize_transform(original_img[i])
                cropped_img = resized_img[:, best_proposal[1]:best_proposal[3], best_proposal[0]:best_proposal[2]]
                wss_output = wss_pipeline(cropped_img, crop_act_map, file_object)  
                inst_mask = np.zeros((im_h, im_w),np.uint8)
                inst_mask[best_proposal[1]:best_proposal[3],best_proposal[0]:best_proposal[2]] = wss_output
                poly_mask = seg_2_poly(inst_mask)

                object_pseudo_list_per_image.append((original_obj_name, best_proposal, score_max, poly_mask)) 
                vis_det_act(original_img[i], act_map_obj, best_proposal, original_obj_name, nearest_folder+'_'+filename, args.output_dir, prop_boxes, poly_mask)

            if proposal_paths[i] not in objects_dict.keys():
                objects_dict[proposal_paths[i]]= object_pseudo_list_per_image
            else:
                objects_dict[proposal_paths[i]].extend(object_pseudo_list_per_image)

        for (k, v) in objects_dict.items():
            file_name = k.split('/')[-1]
            output_addr = os.path.join(args.output_dir, 'pseudo_labels', file_name)
            _, file_extension = os.path.splitext(k)
            
            if file_extension == '':
                output_addr = output_addr+'_pseudo_label.pkl'
            else:
                output_addr = output_addr.replace(file_extension,'_pseudo_label.pkl')

            if not os.path.isdir(os.path.dirname(output_addr)):
                os.makedirs(os.path.dirname(output_addr))

            with open(output_addr, 'wb') as fp:
                pickle.dump(v, fp)

def main(args, config):   
    
    device = torch.device(args.device)
    cudnn.benchmark = True

    ########################################## Dataset ########################################## 
    print("Creating dataset")
    datasets = [create_dataset('pseudolabel', config, args.root_directory, args.bbox_proposal_addr)]

    data_loader = create_loader(datasets, [None],batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    ########################################## Model Initialization ########################################## 
    print("Creating model......")
    bert_config_path = 'ALBEF/configs/config_bert.json'
    model_path = args.model_path
    img_size = 256
    map_size = 16
    model = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=bert_config_path, img_size=img_size)
    model = model.to(device)

    ########################################## Load the Model ##########################################
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    for key in list(state_dict.keys()): # adjust different names in pretrained checkpoint
        if 'bert' in key:
            encoder_key = key.replace('bert.', '')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]

    print("Start loading form the checkpoint......")
    msg = model.load_state_dict(state_dict,strict=False)
    assert len(msg.missing_keys) == 0

    model.eval()
    block_num = 8

    model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True 
    print("Loading object name dictionary....")
    with open(args.object_dict, 'r') as fp:
        object_name_dict = json.load(fp)
    print("Start generating pseudo-mask annotation (box level + pixel level)...!!!")
    start_time = time.time()
    generate_pseudo_bbox(model, tokenizer, data_loader, object_name_dict, args, block_num, map_size, device)
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='ALBEF/configs/Pretrain.yaml')
    parser.add_argument('--model_path', default='examples/ALBEF.pth')
    parser.add_argument('--root_directory', default='datasets/')
    parser.add_argument('--output_dir', default='pseudo_label_output/')
    parser.add_argument('--object_dict', default='examples/object_vocab.json')
    parser.add_argument('--bbox_proposal_addr', default='examples/proposals/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
