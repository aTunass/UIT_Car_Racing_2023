import torch
import argparse
import json
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.model.unet import UNet


def export_torchscript(weight, model, im):

    print(f'Starting export with torch {torch.__version__}...')
    f = weight.replace('.pt','.torchscript')

    ts = torch.jit.trace(model, im, strict=False)
    d = {'shape': im.shape}
    extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
    
    ts.save(str(f), _extra_files=extra_files)

    return f
    

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    model = UNet()
    checkpoint = torch.load(args.weight, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    im = torch.randn(1, 3, 80, 160).to(device).float()

    for _ in range(2):
        y = model(im)  # dry runs

    path = export_torchscript(args.weight, model, im)
    print("Serializing torchscript to file: {:}".format(path))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default= 'weights/seg.pt', help='model.pt or model.pth path')

    args = parser.parse_args()

    main(args)