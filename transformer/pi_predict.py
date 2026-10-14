import argparse, json, time
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

def softmax(x):
    x=x-np.max(x,axis=1,keepdims=True); e=np.exp(x); return e/np.sum(e,axis=1,keepdims=True)

def prep(path, size, mean, std):
    im=Image.open(path).convert('RGB').resize((size,size), Image.Resampling.BILINEAR)
    x=np.asarray(im,dtype=np.float32)/255.0
    x=np.transpose(x,(2,0,1))
    mean=np.asarray(mean,dtype=np.float32).reshape(3,1,1)
    std=np.asarray(std,dtype=np.float32).reshape(3,1,1)
    return np.expand_dims((x-mean)/std,0).astype(np.float32)

ap=argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--meta', required=True)
ap.add_argument('--image', required=True)
ap.add_argument('--benchmark', type=int, default=20)
a=ap.parse_args()
meta=json.loads(Path(a.meta).read_text())
sess=ort.InferenceSession(a.model, providers=['CPUExecutionProvider'])
name=sess.get_inputs()[0].name
x=prep(a.image,int(meta['img_size']),meta['mean'],meta['std'])
for _ in range(3): sess.run(None,{name:x})
t0=time.perf_counter()
for _ in range(max(1,a.benchmark)): logits=sess.run(None,{name:x})[0]
ms=(time.perf_counter()-t0)/max(1,a.benchmark)*1000
p=softmax(logits)[0]; order=np.argsort(p)[::-1]; classes=meta['class_names']
print('prediction :',classes[int(order[0])])
print('confidence :',f'{p[order[0]]*100:.2f}%')
print('top 3:')
for i in order[:3]: print(' ',classes[int(i)],f'{p[i]*100:.2f}%')
print('latency    :',f'{ms:.2f} ms/image')
print('throughput :',f'{1000/ms:.2f} images/s')
