import argparse, json
from pathlib import Path
import numpy as np
import torch
import config
from model import build_model

ap=argparse.ArgumentParser()
ap.add_argument('--checkpoint', required=True)
ap.add_argument('--out', default='transformer_fault_classifier.onnx')
ap.add_argument('--meta', default='transformer_fault_classifier.json')
ap.add_argument('--verify', action='store_true')
args=ap.parse_args()

ck=torch.load(args.checkpoint, map_location='cpu', weights_only=False)
classes=ck['class_names']
backbone=ck.get('backbone', config.BACKBONE)
model=build_model(num_classes=len(classes), backbone=backbone, pretrained=False)
model.load_state_dict(ck['state_dict'])
model.eval()
dummy=torch.randn(1,3,config.IMG_SIZE,config.IMG_SIZE)

torch.onnx.export(model,dummy,args.out,export_params=True,opset_version=18,
                  do_constant_folding=True,input_names=['input'],output_names=['logits'],
                  dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}},dynamo=False)

Path(args.meta).write_text(json.dumps({
    'class_names':list(classes),'img_size':int(config.IMG_SIZE),
    'mean':list(config.MEAN),'std':list(config.STD),'backbone':backbone,
    'input_name':'input','output_name':'logits'
}, indent=2))
print('Exported:', Path(args.out).resolve())
print('Metadata:', Path(args.meta).resolve())

if args.verify:
    import onnxruntime as ort
    with torch.no_grad(): pt=model(dummy).numpy()
    sess=ort.InferenceSession(args.out, providers=['CPUExecutionProvider'])
    ox=sess.run(None,{sess.get_inputs()[0].name:dummy.numpy().astype(np.float32)})[0]
    print('max abs difference:', float(np.max(np.abs(pt-ox))))
    print('same predicted class:', bool(np.argmax(pt,1)[0]==np.argmax(ox,1)[0]))
