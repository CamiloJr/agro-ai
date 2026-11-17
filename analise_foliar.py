"""Processamento de imagem explicavel para evidenciar padroes cromaticos.

Este modulo NAO diagnostica doencas. Mascara HSV = heuristica de cores,
nao segmentacao supervisionada/validada de lesoes ou de uma folha.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class Evidencias:
    folha: np.ndarray
    amarelado: np.ndarray
    escurecido: np.ndarray
    areas_marcadas: np.ndarray
    hsv: np.ndarray
    percent_amarelo: float
    percent_escuro: float
    percent_marcado: float
    region_pixels: int
    nitidez: float
    brilho: float
    qualidade: list[str]
    criterio: str


def _filter(binary: np.ndarray, size=3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    m = cv2.morphologyEx(binary.astype(np.uint8)*255, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)


def detect_region(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Regiao de observacao estimada: pixels coloridos no centro, se robustos.

    Retorna toda a ROI quando a segmentacao simples nao e confiavel. A regiao
    estimada nao deve ser apresentada como contorno botanico da folha.
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:,:,1]
    value = hsv[:,:,2]
    # Em fotos com fundo neutro destaca objetos cromaticamente relevantes.
    candidate = (sat >= 43) & (value >= 33) & (value <= 252)
    m = _filter(candidate, 5)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours,key=cv2.contourArea)
        coverage = cv2.contourArea(largest) / (h*w)
        x,y,bw,bh = cv2.boundingRect(largest)
        center_x = x+bw/2
        center_y = y+bh/2
        if 0.06 < coverage < 0.95 and abs(center_x-w/2)<w*.42 and abs(center_y-h/2)<h*.42:
            out=np.zeros((h,w),np.uint8)
            cv2.drawContours(out,[largest],-1,255,-1)
            return out,'Regiao cromatica aproximada (HSV)'
    return np.full((h,w),255,dtype=np.uint8),'Regiao inteira da imagem (ROI)'


def calculate_evidence(rgb: np.ndarray, sensitivity: float=0.50) -> Evidencias:
    if rgb.ndim!=3 or rgb.shape[2]!=3 or not rgb.size:
        raise ValueError('A imagem deve ser RGB, nao vazia.')
    sensitivity=max(0.,min(1.,float(sensitivity)))
    if rgb.dtype!=np.uint8:
        rgb=np.clip(rgb,0,255).astype(np.uint8)
    hsv=cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(hsv)
    regiao, criterio=detect_region(rgb)
    region=regiao>0
    # Limiar regulavel: so muda a VISUALIZACAO cromatica, nunca a predicao da RNA.
    saturation_limit=int(85-45*sensitivity)
    yellow=(H>=14)&(H<=40)&(S>=saturation_limit)&(V>=75)&region
    # Marrons e necroses escuras (aproximacao). Nao especifica patogenos.
    brown=(H>=2)&(H<=27)&(S>=saturation_limit)&(V>=20)&(V<=int(150+40*sensitivity))&region
    dark=(V<=int(55+38*sensitivity))&(S>=25)&region
    yellow_mask=_filter(yellow,3)
    brown_mask=_filter(brown|dark,3)
    # Evita dupla contagem nas areas evidenciadas.
    brown_mask=cv2.bitwise_and(brown_mask,cv2.bitwise_not(yellow_mask))
    union=cv2.bitwise_or(yellow_mask,brown_mask)
    area=max(1,int(np.count_nonzero(region)))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    sharpness=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    brightness=float(V[region].mean()) if region.any() else float(V.mean())
    warnings=[]
    if brightness<65:warnings.append('Pouca iluminacao')
    if brightness>225:warnings.append('Possivel superexposicao')
    if sharpness<45:warnings.append('Possivel desfoque')
    if criterio.startswith('Regiao inteira'):warnings.append('Fundo/folha nao separados')
    return Evidencias(regiao,yellow_mask,brown_mask,union,hsv,
                     100*np.count_nonzero(yellow_mask)/area,
                     100*np.count_nonzero(brown_mask)/area,
                     100*np.count_nonzero(union)/area,area,
                     sharpness,brightness,warnings,criterio)


def overlay_evidence(rgb:np.ndarray,ev:Evidencias,opacity:float=.60,view='evidencias')->np.ndarray:
    opacity=max(0,min(1,float(opacity)))
    base=rgb.copy()
    if view=='original':return base
    if view=='mapa':
        base=np.zeros_like(rgb)
        base[ev.folha>0]=(33,65,52)
    color=np.zeros_like(rgb)
    color[ev.amarelado>0]=(255,213,52)
    color[ev.escurecido>0]=(244,88,103)
    points=ev.areas_marcadas>0
    base[points]=np.uint8((1-opacity)*base[points].astype(float)+opacity*color[points].astype(float))
    if view=='evidencias':
        # Contorno aproximado apenas da regiao que participa do calculo.
        contours,_=cv2.findContours(ev.folha,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base,contours,-1,(64,234,158),1,lineType=cv2.LINE_AA)
    return base


def crop_roi(image:np.ndarray,roi:tuple[int,int,int,int]|None)->np.ndarray:
    if roi is None:return image
    x1,y1,x2,y2=map(int,roi)
    h,w=image.shape[:2]
    x1,x2=sorted((max(0,min(w,x1)),max(0,min(w,x2))))
    y1,y2=sorted((max(0,min(h,y1)),max(0,min(h,y2))))
    if x2-x1<24 or y2-y1<24:raise ValueError('ROI pequena: selecione uma area maior.')
    return image[y1:y2,x1:x2]


def image_aspect_rect(w:int,h:int,box:tuple[int,int,int,int])->tuple[int,int,int,int]:
    bx,by,bw,bh=box
    s=min(bw/max(1,w),bh/max(1,h))
    nw,nh=max(1,round(w*s)),max(1,round(h*s))
    return (bx+(bw-nw)//2,by+(bh-nh)//2,nw,nh)
