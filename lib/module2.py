import os
import torch.nn.functional as F
import torch
from torch_geometric.nn import GATConv
from openpyxl import load_workbook
from collections import Counter

class Net(torch.nn.Module):
    def __init__(self, in_size=16, hid_size=8, out_size=2):
        super(Net, self).__init__()
        self.conv1 = GATConv(in_size, hid_size, heads=1, concat=False)
        self.conv2 = GATConv(hid_size, out_size, heads=1, concat=False)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x_emb = self.conv1(x, edge_index, edge_weight)
        # print("x_emb", x_emb.size(),"edge_index",edge_index.size(), "edge_weight",edge_weight.size())
        x = F.relu(x_emb)
        x = F.dropout(x, training=self.training)
        # print("x", x.size(),"edge_index",edge_index.size(), "edge_weight",edge_weight.size())
        x = self.conv2(x, edge_index, edge_weight)
        
        return x, x_emb


