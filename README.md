# AngioGraphCAD

Official implementation of the paper "Future cardiovascular events prediction from invasive coronary angiography: A graph representation learning perspective", an extended version of our conference paper titled [Graph Neural Network based Future Clinical Events Prediction from Invasive Coronary Angiography](https://ieeexplore.ieee.org/abstract/document/10635813)
 which was presented at IEEE International Symposium on Biomedical Imaging (ISBI), 2024.



training and validation curve can be found here:
https://wandb.ai/xsun/FAME2MI/reports/Untitled-Report--Vmlldzo5NDA4NDI3

1. Graph construction pipeline 
graph.ipynb shows an example of how to construct lesion graph from an ICA image. 
use torch_geometric.data.Data save the graph into pytorch for training/testing/validation dataloader 
