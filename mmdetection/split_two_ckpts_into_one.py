import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-inp', type=str, required=True)
parser.add_argument('-out', type=str, required=True)
args = parser.parse_args()

two = torch.load(args.inp)
a = {"meta":two["meta"], "state_dict":dict()}
b = {"meta":two["meta"], "state_dict":dict()}
for key, value in two["state_dict"].items():
    key_items = key.split('.')
    if key_items[1].endswith("1"):
        key_items[1] = key_items[1][:-1]
        a["state_dict"]['.'.join(key_items)] = value
    if key_items[1].endswith("2"):
        key_items[1] = key_items[1][:-1]
        b["state_dict"]['.'.join(key_items)] = value

iteration = (args.inp).split('/')[-1][:-4]
torch.save(a, "{}/{}".format(args.out, iteration+".a.pth"))
torch.save(b, "{}/{}".format(args.out, iteration+".b.pth"))
