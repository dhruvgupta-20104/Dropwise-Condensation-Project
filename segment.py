# import torch
# from segment_anything import sam_model_registry

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# MODEL_TYPE = "vit_h"

# sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
# sam.to(device=DEVICE)

# from huggingface_hub import hf_hub_download

# chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
# print(chkpt_path)

# /Users/hirvapatel/.cache/huggingface/hub/models--ybelkada--segment-anything/snapshots/7790786db131bcdc639f24a915d9f2c331d843ee/checkpoints/sam_vit_b_01ec64.pth
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import torch
# import sys
# sys.path.append("..")
# from segment_anything import sam_model_registry, SamPredictor

# sam_checkpoint = "/Users/hirvapatel/Downloads/sam_vit_h_4b8939.pth"
# model_type = "vit_h"

# device = "cuda"

# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

# predictor = SamPredictor(sam)

image = cv2.imread('Dropwise-Condensation-Project/Filtered_Frames/Frame_16.png')
# cv2.imshow('im', image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    print(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
import numpy as np 

# input_point = np.array([[500, 375]])
# input_label = np.array([1])

# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True,
# )
# print(masks.shape)


# def add_SAM_mask_to_detection(detection, mask, img_width, img_height):
#     y0, x0, y1, x1 = fo_to_sam(detection.bounding_box, img_width, img_height)    
#     mask_trimmed = mask[x0:x1+1, y0:y1+1]
#     detection["mask"] = np.array(mask_trimmed)
#     return detection

# def add_SAM_instance_segmentation(sample):
#     w, h = sample.metadata.width, sample.metadata.height
#     image = np.array(PIL.Image.open(sample.filepath))
#     predictor.set_image(image)
    
#     if sample.detections is None:
#         return

#     dets = sample.detections.detections
#     boxes = [d.bounding_box for d in dets]
#     sam_boxes = np.array([fo_to_sam(box, w, h) for box in boxes])
    
#     input_boxes = torch.tensor(sam_boxes, device=predictor.device)
#     transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    
#     masks, _, _ = predictor.predict_torch(
#             point_coords=None,
#             point_labels=None,
#             boxes=transformed_boxes,
#             multimask_output=False,
#         )
    
#     new_dets = []
#     for i, det in enumerate(dets):
#         mask = masks[i, 0]
#         new_dets.append(add_SAM_mask_to_detection(det, mask, w, h))

#     sample.detections = fo.Detections(detections = new_dets)

# def add_SAM_instance_segmentations(dataset):
#     for sample in dataset.iter_samples(autosave=True, progress=True):
#         add_SAM_instance_segmentation(sample)

from segment_anything import build_sam, SamPredictor 
# input_box = np.array([1123.0, 589.0, 1194.0, 678.0])
input_point = np.array([[1158.5, 633.5]])
input_label = np.array([1])
# print(image)

# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()  

predictor = SamPredictor(build_sam(checkpoint="/Users/hirvapatel/Downloads/sam_vit_h_4b8939.pth"))
predictor.set_image(image)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

mask = masks[0]
score = scores[0]
pixels = cv2.countNonZero(mask)
print(pixels)
# for i, (mask, score) in enumerate(zip(masks, scores)):
plt.figure(figsize=(10,10))
plt.imshow(image)
show_mask(mask, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.title(f"Mask Score: {score:.3f}", fontsize=18)
plt.axis('off')
plt.show() 
# masks, _, _ = predictor.predict(point_coords=None,
#     point_labels=None,
#     box=input_box[None,:],
#     multimask_output=False,)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
# # show_points(input_point, input_label, plt.gca())
# plt.axis('off')
# plt.show()