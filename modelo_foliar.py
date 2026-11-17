"""Classificador ONNX para folhas: pre-treinado; inferencia local CPU.

Modelo: imaflower/plantvillage-mobilenetv3, licenca MIT.
ATENCAO: 15 classes especificas, imagens PlantVillage; nao e classificador
aberto de pragas, nem diagnostico agronomico. Nunca inventa predicoes.
"""
from __future__ import annotations
import hashlib
import os
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

REPO='https://huggingface.co/imaflower/plantvillage-mobilenetv3/resolve/main/'
FILENAMES=('model.onnx','model.onnx.data')
# Ordem verificada no class_names.json do modelo (nao alterar a ordem).
LABELS=[
'Pepper__bell___Bacterial_spot',
'Pepper__bell___healthy',
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Tomato_Bacterial_spot',
'Tomato_Early_blight',
'Tomato_Late_blight',
'Tomato_Leaf_Mold',
'Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite',
'Tomato__Target_Spot',
'Tomato__Tomato_YellowLeaf__Curl_Virus',
'Tomato__Tomato_mosaic_virus',
'Tomato_healthy',
]
SHA_ONNX='c6a962f699ddc820108231b23454b6a4e242407044c2c2514a0f6294d35428fc'
PORTUGUESE={
'Pepper__bell___Bacterial_spot':'Pimentao | mancha bacteriana',
'Pepper__bell___healthy':'Pimentao | folha saudavel',
'Potato___Early_blight':'Batata | pinta-preta (alternariose)',
'Potato___Late_blight':'Batata | requeima',
'Potato___healthy':'Batata | folha saudavel',
'Tomato_Bacterial_spot':'Tomate | mancha bacteriana',
'Tomato_Early_blight':'Tomate | pinta-preta (alternariose)',
'Tomato_Late_blight':'Tomate | requeima',
'Tomato_Leaf_Mold':'Tomate | mofo foliar',
'Tomato_Septoria_leaf_spot':'Tomate | septoriose',
'Tomato_Spider_mites_Two_spotted_spider_mite':'Tomate | dano associado a acaros',
'Tomato__Target_Spot':'Tomate | mancha-alvo',
'Tomato__Tomato_YellowLeaf__Curl_Virus':'Tomate | virus do enrolamento amarelo',
'Tomato__Tomato_mosaic_virus':'Tomate | virus do mosaico',
'Tomato_healthy':'Tomate | folha saudavel',
}


def label_pt(raw:str)->str:
    return PORTUGUESE.get(raw,raw.replace('___',' | ').replace('_',' '))


def crop_square_resize(rgb:np.ndarray,side:int)->np.ndarray:
    """Resize da menor dimensao para 'side' e crop central, como no treinamento."""
    if side<=0:raise ValueError('Resolucao de entrada invalida')
    pil=Image.fromarray(rgb.astype(np.uint8),'RGB')
    w,h=pil.size
    scale=side/min(w,h)
    new_w=max(side,round(w*scale))
    new_h=max(side,round(h*scale))
    pil=pil.resize((new_w,new_h),Image.Resampling.BILINEAR)
    x=(new_w-side)//2
    y=(new_h-side)//2
    return np.asarray(pil.crop((x,y,x+side,y+side)),dtype=np.uint8)


def to_tensor(rgb:np.ndarray,side:int=224)->np.ndarray:
    image=crop_square_resize(rgb,side).astype(np.float32)/255.0
    mean=np.array([0.485,0.456,0.406],dtype=np.float32)
    std=np.array([0.229,0.224,0.225],dtype=np.float32)
    return ((image-mean)/std).transpose(2,0,1)[None].copy().astype(np.float32)


def softmax(logits:np.ndarray)->np.ndarray:
    a=np.asarray(logits,dtype=np.float64).reshape(-1)
    if a.size!=len(LABELS):raise ValueError(f'Modelo retornou {a.size} classes; esperado: {len(LABELS)}')
    if not np.isfinite(a).all():raise ValueError('Modelo retornou valores invalidos')
    exp=np.exp(a-np.max(a))
    return exp/np.sum(exp)


def download_weights(folder:Path,progress=None)->Path:
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    for filename in FILENAMES:
        dest=folder/filename
        if dest.is_file() and dest.stat().st_size>(100_000 if filename.endswith('.onnx') else 2_000_000):
            if filename!='model.onnx' or hashlib.sha256(dest.read_bytes()).hexdigest()==SHA_ONNX:
                continue
        if progress:progress(f'Baixando {filename} do repositorio oficial...')
        temp=dest.with_suffix(dest.suffix+'.part')
        try:
            req=urllib.request.Request(REPO+filename,headers={'User-Agent':'AgroIA-Folhas/1.0 educational'})
            with urllib.request.urlopen(req,timeout=45) as source,open(temp,'wb') as out:
                while True:
                    chunk=source.read(1024*256)
                    if not chunk:break
                    out.write(chunk)
            if temp.stat().st_size<(100_000 if filename.endswith('.onnx') else 2_000_000):
                raise RuntimeError(f'Arquivo {filename} incompleto')
            if filename=='model.onnx' and hashlib.sha256(temp.read_bytes()).hexdigest()!=SHA_ONNX:
                raise RuntimeError('O modelo ONNX nao passou na verificacao de integridade (SHA256).')
            os.replace(temp,dest)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise RuntimeError(f'Falha ao baixar {filename}: {exc}.\n'
                               f'Baixe manualmente de {REPO} e salve na pasta modelos.') from exc
    return folder/'model.onnx'


class LeafClassifier:
    def __init__(self,folder:Path):
        self.folder=Path(folder)
        self.session=None
        self.name=None
        self.side=224
    def initialize(self,progress=None):
        if self.session is not None:return
        onnx=download_weights(self.folder,progress)
        if progress:progress('Carregando o modelo ONNX em CPU...')
        try:
            import onnxruntime as ort
            sess=ort.InferenceSession(str(onnx),providers=['CPUExecutionProvider'])
            inp=sess.get_inputs()[0]
            shape=inp.shape
            if len(shape)!=4:raise ValueError(f'Formato inesperado: {shape}')
            self.side=int(shape[2]) if isinstance(shape[2],int) else 224
            if isinstance(shape[3],int) and shape[3]!=self.side:
                raise ValueError('Modelo nao possui entrada quadrada')
            self.name=inp.name
            self.session=sess
        except Exception as exc:
            raise RuntimeError('Falha ao carregar modelo ONNX. Confira se os arquivos '
                               f'model.onnx e model.onnx.data estao juntos: {exc}') from exc
    def predict(self,rgb:np.ndarray,progress=None)->list[dict]:
        self.initialize(progress)
        tensor=to_tensor(rgb,self.side)
        scores=self.session.run(None,{self.name:tensor})[0]
        probs=softmax(scores)
        idxs=np.argsort(probs)[-3:][::-1]
        return [{'raw':LABELS[int(i)],'label':label_pt(LABELS[int(i)]),
                 'confidence':float(probs[int(i)])} for i in idxs]
