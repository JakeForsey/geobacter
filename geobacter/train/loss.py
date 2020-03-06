from torch.functional import F
from torch import nn
import torch


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # anchor_to_positive = F.pairwise_distance(anchor, positive).sum
        # anchor_to_negative = F.pairwise_distance(anchor, negative)
        #
        # positive_to_negative = F.pairwise_distance(positive, negative)
        # # Select the hardest distance to minimize
        # to_negative = torch.min(anchor_to_negative, positive_to_negative)
        #
        # dist = anchor_to_positive.pow(2) - to_negative.pow(2)
        #
        # return torch.mean(F.relu(dist + self.margin))

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()  # if size_average else losses.sum()