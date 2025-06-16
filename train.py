'''
Author: xuexufeng
Date: 2025-02-11 11:51:21
LastEditors: xuexufeng
LastEditTime: 2025-06-16 11:59:19
FilePath: /ultralytics/train.py
Description: 

Copyright (c) 2025 by xuexufeng@pointspread.tech, All Rights Reserved. 
'''
import os
import sys
import numpy as np
import ultralytics.data.build as build
from ultralytics.data import YOLODataset

sys.path.append(os.getcwd())
from ultralytics import YOLO

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
    
    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
              weights.append(1)
              continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

# Load a model
model = YOLO("yolov11-custom.yaml")  # load a pretrained model (recommended for training)
# model.load("yolo11x.pt")
model.load("runs/detect/finetune01/weights/best.pt")
# build.YOLODataset = YOLOWeightedDataset

results = model.train(data="data.yaml", 
                      epochs=5, 
                      imgsz=640, 
                      device=[6,7], 
                      batch=64, 
                      lr0=0.00001,
                      freeze=11, 
                      optimizer="SGD",
                      perspective=0.0005,
                    )

