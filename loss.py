import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss() # only have one class, so no need to specify class
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1 # class loss
        self.lambda_noobj = 10 # no object
        self.lambda_obj = 1 # object
        self.lambda_box = 10 # bounding box

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1 
        noobj = target[..., 0] == 0

        # ======================== #
        #   FOR NO OBJECT LOSS     #
        # ======================== #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]) # no object loss
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2) # 3 anchors for each scale and 2 coordinates for each anchor, p_w * exp(t_w), p_h * exp(t_h), p_w and p_h are the anchors' width and height
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1) # x, y, w, h -> x, y, w, h

        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach() # we detach because we don't want to backpropagate through ious
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj]) # target[..., 0:1][obj] is the IOU

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )

        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj]) 

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()) 
        )

        #print(no_object_loss, object_loss, box_loss, class_loss)
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )