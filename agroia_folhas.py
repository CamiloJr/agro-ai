"""AGRO IA | Laboratorio Interativo de Inspecao Foliar - Aula 7.

Inferencia local ONNX (15 classes) + analise HSV explicavel (nao IA).
Inicie por INICIAR_WINDOWS.bat no Windows ou `python agroia_folhas.py`.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from functools import lru_cache
from pathlib import Path

try:
    import cv2
    import numpy as np
    import pygame
    from PIL import Image, ImageOps
except ImportError as exc:
    raise SystemExit('Biblioteca ausente: execute INICIAR_WINDOWS.bat. Detalhe: '+str(exc))

from analise_foliar import calculate_evidence, crop_roi, overlay_evidence, image_aspect_rect
from modelo_foliar import LeafClassifier

ROOT=Path(__file__).resolve().parent
SCREEN_W,SCREEN_H=1536,960
FPS=45
MAIN_BOX=pygame.Rect(22,151,1015,606)
SIDE_X,SIDE_W=1055,459
BG=(7,17,22)
PANEL=(15,30,37)
INNER=(20,40,48)
LINE=(40,68,73)
TEXT=(236,248,244)
MUTED=(146,173,174)
GREEN=(83,228,155)
TEAL=(65,191,205)
YELLOW=(255,206,91)
RED=(255,121,128)
BLUE=(100,167,255)


@lru_cache(maxsize=96)
def font(size:int,bold:bool=False):
    family=pygame.font.match_font('segoeui') or pygame.font.match_font('arial')
    f=pygame.font.Font(family,int(size))
    f.set_bold(bold)
    return f


def label(surf,content,x,y,size=16,color=TEXT,bold=False,anchor='topleft'):
    image=font(size,bold).render(str(content),True,color)
    r=image.get_rect(**{anchor:(round(x),round(y))})
    surf.blit(image,r)
    return r


def panel(surf,rect,fill=PANEL,border=LINE,radius=15):
    r=pygame.Rect(rect)
    pygame.draw.rect(surf,fill,r,border_radius=radius)
    if border:pygame.draw.rect(surf,border,r,1,border_radius=radius)
    return r


def truncate(value,max_width,font_size=14):
    value=str(value)
    if font(font_size).size(value)[0]<=max_width:return value
    while value and font(font_size).size(value+'…')[0]>max_width:
        value=value[:-1]
    return value+'…'


def wrapped(text,px_width=430,max_rows=3,size=14):
    result=[]
    for para in str(text).split('\n'):
        buffer=''
        for word in para.split():
            candidate=(buffer+' '+word).strip()
            if font(size).size(candidate)[0]>px_width and buffer:
                result.append(buffer)
                buffer=word
            else:buffer=candidate
        if buffer:result.append(buffer)
        if len(result)>=max_rows:return result[:max_rows]
    return result[:max_rows]


class InferenceThread(threading.Thread):
    """Fila de inferencia: nao bloqueia o loop grafico durante downloads/rede."""
    def __init__(self,model_dir:Path):
        super().__init__(daemon=True)
        self.jobs=queue.Queue(maxsize=1)
        self.responses=queue.Queue()
        self.classifier=LeafClassifier(model_dir)
        self.busy=False
        self.message='Modelo aguardando imagem'
        self.start()

    def submit(self,job_id:int,rgb:np.ndarray):
        if self.busy or not self.jobs.empty():return False
        self.busy=True
        self.jobs.put((job_id,rgb.copy()))
        return True

    def run(self):
        while True:
            job_id,image=self.jobs.get()
            try:
                self.message='Preparando modelo e inferencia...'
                preds=self.classifier.predict(image,progress=lambda msg:setattr(self,'message',msg))
                self.responses.put((job_id,preds,None))
                self.message='Modelo carregado | inferencia local'
            except Exception as exc:
                self.responses.put((job_id,None,str(exc)))
                self.message='Falha na inferencia; confira o log'
                with (ROOT/'agroia_erros.log').open('a',encoding='utf-8') as log:
                    log.write('\n### '+datetime.now().isoformat()+' ###\n'+traceback.format_exc())
            finally:
                self.busy=False
                self.jobs.task_done()


class AgroIALab:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('AGRO IA | Inspecao Foliar | Inteligencia Artificial - Aula 7')
        self.screen=pygame.display.set_mode((SCREEN_W,SCREEN_H),pygame.RESIZABLE)
        self.canvas=pygame.Surface((SCREEN_W,SCREEN_H)).convert()
        self.clock=pygame.time.Clock()
        self.worker=InferenceThread(ROOT/'modelos')
        self.running=True
        self.source='vazio'  # vazio, camera, imagem
        self.camera=None
        self.camera_index=0
        self.frame=None
        self.file_path=None
        self.predictions=[]
        self.pred_error=None
        self.evidence=None
        self.roi=None
        self.roi_edit=False
        self.drag_start=None
        self.drag_now=None
        self.image_rect=None
        self.buttons=[]
        self.view='evidencias'
        self.sensitivity=.50
        self.opacity=.57
        self.min_confidence=.55
        self.help_visible=False
        self.status='Abra uma imagem ou inicie a webcam para analisar uma folha.'
        self.status_time=time.monotonic()
        self.timestamp_last_submit=0.
        self.timestamp_last_evidence=0.
        self.image_id=0
        self.generation=0  # avanca quando a imagem ou regiao de analise muda
        self.request_generation=-1
        self.needs_inference=False
        self.last_prediction_id=-1
        self.last_inference_ms=None
        self.request_start=None
        self.last_export=''
        self.last_canvas_frame=None
        self.camera_fps=0
        self.last_cam_timestamp=0
        self.frame_num=0

    def message(self,msg):
        self.status=str(msg)
        self.status_time=time.monotonic()

    def real_to_virtual(self,pos):
        w,h=self.screen.get_size()
        k=min(w/SCREEN_W,h/SCREEN_H)
        dx=(w-SCREEN_W*k)/2
        dy=(h-SCREEN_H*k)/2
        return ((pos[0]-dx)/k,(pos[1]-dy)/k)

    def draw_button(self,surf,title,rect,callback,active=False,color=None,fontsize=14,disabled=False):
        r=pygame.Rect(rect) # suporte tanto a tuplas quanto Rects
        mouse=self.real_to_virtual(pygame.mouse.get_pos())
        hover=r.collidepoint(mouse) and not disabled
        bg=(29,92,68) if active else ((38,65,71) if hover else (23,47,53))
        edge=GREEN if active else ((83,137,139) if hover else LINE)
        fg=TEXT if not disabled else (87,113,114)
        if color and not disabled:
            bg=color if active else (28,60,63)
        pygame.draw.rect(surf,bg,r,border_radius=9)
        pygame.draw.rect(surf,edge,r,1,border_radius=9)
        label(surf,title,r.centerx,r.centery,fontsize,fg,active or hover,'center')
        if not disabled:self.buttons.append((r,callback))

    def increment(self,which,amount,lo,hi):
        value=max(lo,min(hi,getattr(self,which)+amount))
        setattr(self,which,round(value,2))
        if which=='sensitivity':self.refresh_evidence()

    def draw_adjuster(self,surf,x,y,name,field,formatter,step,lo,hi):
        label(surf,name,x,y,13,MUTED)
        self.draw_button(surf,'−',(x,y+20,34,31),lambda:self.increment(field,-step,lo,hi),fontsize=19)
        label(surf,formatter.format(getattr(self,field)),x+103,y+36,16,TEXT,True,'center')
        self.draw_button(surf,'+',(x+184,y+20,34,31),lambda:self.increment(field,step,lo,hi),fontsize=19)

    def stop_camera(self):
        if self.camera is not None:
            try:self.camera.release()
            except Exception:pass
        self.camera=None
        if self.source=='camera':
            self.source='vazio' if self.frame is None else 'imagem'

    def toggle_camera(self):
        if self.source=='camera':
            self.freeze_frame()
            return
        self.stop_camera()
        self.roi=None
        cap=None
        indices=list(dict.fromkeys([self.camera_index,0,1,2]))
        apis=([cv2.CAP_DSHOW,cv2.CAP_MSMF,cv2.CAP_ANY]
              if os.name=='nt' else [cv2.CAP_ANY])
        for idx in indices:
            for api in apis:
                test=None
                try:
                    test=cv2.VideoCapture(idx,api)
                    if test.isOpened():
                        ok,bgr=test.read()
                        if ok and bgr is not None:
                            cap=test
                            self.camera_index=idx
                            self.frame=cv2.cvtColor(cv2.flip(bgr,1),cv2.COLOR_BGR2RGB)
                            break
                except Exception:
                    pass
                if test is not None:test.release()
            if cap is not None:break
        if cap is None:
            self.message('Webcam indisponivel. Verifique permissao do Windows e outros programas.')
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.camera=cap
        self.source='camera'
        self.file_path=None
        self.roi_edit=False
        self.predictions=[]
        self.pred_error=None
        self.last_prediction_id=-1
        self.generation+=1
        self.needs_inference=False
        self.timestamp_last_submit=0
        self.message('Webcam ativa. A IA faz inferencias periodicas; imagem ao vivo.')
        self.refresh_evidence()

    def freeze_frame(self):
        if self.source!='camera' or self.frame is None:return
        self.stop_camera()
        self.source='imagem'
        self.frame=self.frame.copy()
        self.predictions=[]
        self.pred_error=None
        self.generation+=1
        self.needs_inference=True
        self.message('Quadro capturado. Agora voce pode selecionar uma regiao (ROI).')
        self.refresh_evidence()
        self.schedule_inference(force=True)

    def choose_image(self):
        try:
            from tkinter import Tk, filedialog
            root=Tk()
            root.withdraw()
            root.attributes('-topmost',True)
            path=filedialog.askopenfilename(parent=root,title='Selecione uma imagem de folha',
                filetypes=[('Imagens','*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff'),('Todos','*.*')])
            root.destroy()
        except Exception as exc:
            self.message(f'Nao foi possivel abrir o seletor de arquivos: {exc}')
            return
        if path:
            try:self.load_image(Path(path))
            except Exception as exc:self.message(f'Nao foi possivel ler a imagem: {exc}')

    def load_image(self,path:Path):
        path=Path(path)
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            # Limita memoria de fotos muito grandes, preservando proporcao.
            im.thumbnail((2200,1600),Image.Resampling.LANCZOS)
            rgb=np.asarray(im,dtype=np.uint8).copy()
        if rgb.size==0:raise ValueError('Arquivo de imagem vazio.')
        self.stop_camera()
        self.frame=rgb
        self.source='imagem'
        self.file_path=path
        self.roi=None
        self.roi_edit=False
        self.predictions=[]
        self.pred_error=None
        self.last_prediction_id=-1
        self.generation+=1
        self.needs_inference=True
        self.message('Imagem carregada. Executando analise da folha...')
        self.refresh_evidence()
        self.schedule_inference(force=True)

    def sample_area(self):
        if self.frame is None:return None
        return crop_roi(self.frame,self.roi)

    def refresh_evidence(self):
        if self.frame is None:
            self.evidence=None
            return
        try:
            area=self.sample_area()
            # Thumbnail para nao bloquear a GUI com imagens grandes.
            h,w=area.shape[:2]
            factor=min(1,850/max(w,h))
            if factor<1:
                area=cv2.resize(area,(max(1,round(w*factor)),max(1,round(h*factor))),interpolation=cv2.INTER_AREA)
            self.evidence=calculate_evidence(area,self.sensitivity)
            self.timestamp_last_evidence=time.monotonic()
        except Exception as exc:
            self.evidence=None
            self.message('Falha na analise cromatica: '+str(exc))

    def schedule_inference(self,force=False):
        if self.frame is None:return
        if self.worker.busy:return
        now=time.monotonic()
        if not force and now-self.timestamp_last_submit < (2.0 if self.source=='camera' else 100):return
        area=self.sample_area()
        # Limita o tamanho antes de enviar para thread.
        h,w=area.shape[:2]
        if max(h,w)>1400:
            ratio=1400/max(h,w)
            area=cv2.resize(area,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_AREA)
        self.image_id+=1
        jid=self.image_id
        if self.worker.submit(jid,area):
            self.request_generation=self.generation
            self.needs_inference=False
            self.timestamp_last_submit=now
            self.request_start=now
            self.pred_error=None
            if self.source=='imagem':self.message('Executando classificador ONNX...')

    def update(self,dt):
        if self.source=='camera' and self.camera is not None:
            ok,bgr=self.camera.read()
            if ok and bgr is not None:
                self.frame=cv2.cvtColor(cv2.flip(bgr,1),cv2.COLOR_BGR2RGB)
                now=time.monotonic()
                self.camera_fps=1/max(1e-4,now-self.last_cam_timestamp) if self.last_cam_timestamp else 0
                self.last_cam_timestamp=now
                if now-self.timestamp_last_evidence>.7:self.refresh_evidence()
                self.schedule_inference()
            else:
                self.message('Webcam desconectada. Verifique o dispositivo.')
                self.stop_camera()
        while True:
            try:jid,preds,err=self.worker.responses.get_nowait()
            except queue.Empty:break
            # Nao apresentar resultados referentes a imagem/ROI antiga.
            if jid==self.image_id and self.request_generation==self.generation:
                self.predictions=preds or []
                self.pred_error=err
                self.last_prediction_id=jid
                self.last_inference_ms=(time.monotonic()-self.request_start)*1000 if self.request_start else None
                if err:self.message('Falha na IA: '+err[:180])
                elif self.predictions:
                    self.message('Inferencia concluida; resultados indicativos, nao diagnostico.')
            self.worker.responses.task_done()
        if self.source=='imagem' and self.needs_inference and not self.worker.busy:
            self.schedule_inference(force=True)

    def set_view(self,v):self.view=v

    def roi_on(self):
        if self.frame is None:
            self.message('Abra uma imagem antes de selecionar a folha.')
            return
        if self.source=='camera':
            self.message('Capture um quadro antes de selecionar a ROI.')
            return
        self.roi_edit=not self.roi_edit
        self.message('Arraste um retangulo sobre a folha para definir a area de interesse.' if self.roi_edit
                     else 'Selecao de ROI desativada.')

    def clear_roi(self):
        if self.frame is None:return
        self.roi=None
        self.roi_edit=False
        self.predictions=[]
        self.generation+=1
        self.needs_inference=True
        self.refresh_evidence()
        self.schedule_inference(force=True)
        self.message('ROI limpa. Analise da imagem completa solicitada.')

    def pointer_to_image(self,p):
        if self.frame is None or self.image_rect is None:return None
        r=self.image_rect
        if not r.collidepoint(p):return None
        h,w=self.frame.shape[:2]
        ix=min(w-1,max(0,int((p[0]-r.x)*w/r.w)))
        iy=min(h-1,max(0,int((p[1]-r.y)*h/r.h)))
        return ix,iy

    def set_roi_drag(self,first,second):
        if first is None or second is None:return
        x1,x2=sorted([first[0],second[0]])
        y1,y2=sorted([first[1],second[1]])
        if x2-x1<24 or y2-y1<24:
            self.message('Area pequena. Arraste para selecionar uma folha maior.')
            return
        self.roi=(x1,y1,x2,y2)
        self.roi_edit=False
        self.predictions=[]
        self.generation+=1
        self.needs_inference=True
        self.refresh_evidence()
        self.schedule_inference(force=True)
        self.message('ROI definida. A IA analisara apenas a regiao selecionada.')

    def render_frame(self)->np.ndarray | None:
        if self.frame is None:return None
        if self.view=='original' or self.evidence is None:return self.frame
        area=self.sample_area()
        h,w=area.shape[:2]
        ah,aw=self.evidence.folha.shape
        # Exibe a mesma area analisada, redimensionando a mascara ao tamanho original.
        if (w,h)!=(aw,ah):
            scaled_area=cv2.resize(area,(aw,ah),interpolation=cv2.INTER_AREA)
        else:scaled_area=area
        ov=overlay_evidence(scaled_area,self.evidence,self.opacity,
            view='mapa' if self.view=='mapa' else 'evidencias')
        if ov.shape[:2]!=(h,w):ov=cv2.resize(ov,(w,h),interpolation=cv2.INTER_LINEAR)
        whole=self.frame.copy()
        if self.roi is None:whole=ov
        else:
            x1,y1,x2,y2=self.roi
            whole[y1:y2,x1:x2]=ov
        return whole

    def export(self):
        if self.frame is None:
            self.message('Nenhuma imagem disponivel para exportar.')
            return
        out=ROOT/'relatorios'
        out.mkdir(exist_ok=True)
        now=datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix=f'agroia_{now}'
        img_path=out/(suffix+'.png')
        json_path=out/(suffix+'.json')
        picture=self.render_frame()
        Image.fromarray(picture,'RGB').save(img_path)
        info={
          'programa':'AGRO IA Folhas - Aula 7',
          'data':datetime.now().isoformat(timespec='seconds'),
          'origem':str(self.file_path) if self.file_path else self.source,
          'modelo':'imaflower/plantvillage-mobilenetv3 (ONNX, 15 classes)',
          'predicoes_top3':self.predictions,
          'limiar_de_confianca':self.min_confidence,
          'roi_px':self.roi,
          'analise_visual_heuristica':None if self.evidence is None else {
              'amarelamento_porcentagem_aproximada':round(self.evidence.percent_amarelo,2),
              'escurecimento_porcentagem_aproximada':round(self.evidence.percent_escuro,2),
              'marcacoes_porcentagem_aproximada':round(self.evidence.percent_marcado,2),
              'criterio_regiao':self.evidence.criterio,
              'avisos_de_qualidade':self.evidence.qualidade,
          },
          'aviso':'Triagem educacional; classe e mascaras nao constituem diagnostico fitossanitario.\n'
                 'O classificador nao identifica pragas individuais nem estima severidade de doencas.'
        }
        json_path.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
        self.last_export=str(img_path)
        self.message('PNG e JSON salvos em relatorios/: '+img_path.name)

    def draw_header(self,surf):
        label(surf,'AGRO',25,17,32,GREEN,True)
        label(surf,'IA',133,17,32,TEXT,True)
        label(surf,'LABORATORIO DE INSPECAO FOLIAR',25,61,14,MUTED,True)
        panel(surf,(520,20,430,65),(17,48,47),LINE,12)
        label(surf,'CLASSIFICACAO POR IA  +  ANALISE CROMATICA',538,32,15,GREEN,True)
        label(surf,'Duas tecnicas distintas para comparar em aula',538,57,12,MUTED)
        panel(surf,(1060,20,450,65),PANEL,LINE,12)
        label(surf,'MODO DIDATICO  •  NAO E DIAGNOSTICO',1080,33,15,YELLOW,True)
        label(surf,'Tomate  |  Batata  |  Pimentao',1080,59,13,MUTED)
        pygame.draw.line(surf,LINE,(22,111),(1513,111),1)

    def draw_image(self,surf):
        r=panel(surf,MAIN_BOX,(12,28,31),LINE,15)
        pygame.draw.rect(surf,(14,38,36),(r.x+1,r.y+1,r.w-2,48),border_top_left_radius=15,border_top_right_radius=15)
        label(surf,'VISOR DE INSPECAO',r.x+19,r.y+16,15,TEXT,True)
        label(surf,'WEBCAM AO VIVO' if self.source=='camera' else ('IMAGEM CARREGADA' if self.frame is not None else 'AGUARDANDO ENTRADA'),
              r.right-18,r.y+17,12,GREEN if self.source=='camera' else MUTED,True,'topright')
        area=pygame.Rect(r.x+13,r.y+59,r.w-26,r.h-74)
        pygame.draw.rect(surf,(8,21,26),area,border_radius=9)
        self.image_rect=None
        if self.frame is None:
            centerx,centery=area.center
            pygame.draw.circle(surf,(30,65,56),(centerx,centery-50),66,3)
            pygame.draw.line(surf,GREEN,(centerx-29,centery-69),(centerx+25,centery-21),5)
            pygame.draw.line(surf,GREEN,(centerx+25,centery-21),(centerx+63,centery-88),5)
            label(surf,'Uma folha, multiplas evidencias',centerx,centery+50,25,TEXT,True,'center')
            label(surf,'Abra uma foto ou inicie a webcam para visualizar a analise',centerx,centery+89,15,MUTED,False,'center')
            return
        pic=self.render_frame()
        h,w=pic.shape[:2]
        rr=pygame.Rect(image_aspect_rect(w,h,(area.x,area.y,area.w,area.h)))
        self.image_rect=rr
        surf_image=pygame.image.fromstring(pic.tobytes(),(w,h),'RGB')
        small=pygame.transform.smoothscale(surf_image,rr.size)
        surf.blit(small,rr)
        pygame.draw.rect(surf,(68,105,94),rr,1)
        if self.roi:
            x1,y1,x2,y2=self.roi
            box=pygame.Rect(rr.x+round(x1*rr.w/w),rr.y+round(y1*rr.h/h),
                            max(2,round((x2-x1)*rr.w/w)),max(2,round((y2-y1)*rr.h/h)))
            pygame.draw.rect(surf,GREEN,box,2)
            label(surf,'ROI',box.x+5,max(rr.y+4,box.y-22),13,GREEN,True)
        if self.drag_start and self.drag_now:
            bx,by=self.drag_start
            ex,ey=self.drag_now
            temp=pygame.Rect(rr.x+min(bx,ex)*rr.w/w,rr.y+min(by,ey)*rr.h/h,
                             abs(bx-ex)*rr.w/w,abs(by-ey)*rr.h/h)
            pygame.draw.rect(surf,YELLOW,temp,2)
        if self.view!='original':
            label(surf,'MAPA DE CORES: HEURISTICA HSV - NAO DIAGNOSTICO',rr.x+11,rr.bottom-25,12,YELLOW,True)
        label(surf,f'{w} x {h} px',area.right-10,area.bottom-9,12,TEAL,True,'bottomright')

    def draw_right(self,surf):
        x=SIDE_X
        y=151
        panel(surf,(x,y,SIDE_W,214),PANEL)
        label(surf,'PREDICAO DA REDE NEURAL',x+17,y+15,16,TEXT,True)
        label(surf,'MobileNetV3 | PlantVillage | 15 classes',x+17,y+41,13,MUTED)
        if self.frame is None:
            label(surf,'Aguardando imagem',x+18,y+94,18,MUTED)
        elif self.pred_error:
            label(surf,'MODELO INDISPONIVEL',x+18,y+75,16,RED,True)
            for n,line in enumerate(wrapped(self.pred_error,420,4,12)):
                label(surf,line,x+18,y+104+n*19,12,MUTED)
        elif not self.predictions:
            label(surf,'PROCESSANDO...' if self.worker.busy else 'Sem previsao',x+18,y+92,16,YELLOW,True)
            for n,line in enumerate(wrapped(self.worker.message,420,2,12)):
                label(surf,line,x+18,y+126+n*19,12,MUTED)
        else:
            top=self.predictions[0]['confidence']
            if top<self.min_confidence:
                label(surf,'INCERTO: revisao necessaria',x+18,y+65,13,YELLOW,True)
            else:label(surf,'HIPOTESES DO MODELO (nao diagnostico)',x+18,y+65,12,GREEN,True)
            for i,p in enumerate(self.predictions[:3]):
                yy=y+87+i*40
                color=GREEN if i==0 else TEAL if i==1 else MUTED
                label(surf,f'{i+1}. '+truncate(p['label'],340,13),x+17,yy,13,TEXT,i==0)
                label(surf,f"{p['confidence']*100:.1f}%",x+SIDE_W-19,yy,13,color,True,'topright')
                pygame.draw.rect(surf,(35,61,62),(x+18,yy+23,421,5),border_radius=3)
                pygame.draw.rect(surf,color,(x+18,yy+23,max(1,int(421*p['confidence'])),5),border_radius=3)

        y=378
        panel(surf,(x,y,SIDE_W,195),PANEL)
        label(surf,'MAPA DE ALTERACOES VISUAIS',x+17,y+15,16,TEXT,True)
        label(surf,'Regra de cor HSV, independente da RNA',x+17,y+41,13,MUTED)
        if self.evidence is None:
            label(surf,'Aguardando imagem',x+19,y+95,15,MUTED)
        else:
            ev=self.evidence
            for i,(name,val,color) in enumerate([('Amarelo/clorose visual',ev.percent_amarelo,YELLOW),
                                                   ('Escuro/marrom visual',ev.percent_escuro,RED),
                                                   ('Pixels destacados',ev.percent_marcado,GREEN)]):
                yy=y+69+i*35
                label(surf,name,x+17,yy,13,MUTED)
                label(surf,f'{val:.1f}%',x+440,yy,14,color,True,'topright')
                pygame.draw.rect(surf,(37,59,60),(x+18,yy+21,420,5),border_radius=3)
                pygame.draw.rect(surf,color,(x+18,yy+21,max(1,int(420*min(100,val)/100)),5),border_radius=3)
        y=586
        panel(surf,(x,y,SIDE_W,178),PANEL)
        label(surf,'QUALIDADE E INTERPRETACAO',x+17,y+15,15,TEXT,True)
        if self.evidence is None:
            label(surf,'Os dados aparecerao apos o carregamento.',x+18,y+59,13,MUTED)
        else:
            ev=self.evidence
            label(surf,f'Nitidez (Laplaciano): {ev.nitidez:.0f}',x+17,y+46,13,MUTED)
            label(surf,f'Brilho medio: {ev.brilho:.0f}/255',x+17,y+68,13,MUTED)
            line=' / '.join(ev.qualidade) if ev.qualidade else 'Sem alertas simples de qualidade'
            for i,l in enumerate(wrapped(line,418,2,12)):
                label(surf,l,x+17,y+91+i*17,12,YELLOW if ev.qualidade else GREEN)
            for i,l in enumerate(wrapped(ev.criterio,418,2,12)):
                label(surf,l,x+17,y+136+i*17,12,MUTED)

    def draw_controls(self,surf):
        panel(surf,(22,771,1015,153),PANEL)
        label(surf,'CONTROLES DE ANALISE',39,784,14,TEXT,True)
        view_names=[('original','Original'),('evidencias','Sintomas (HSV)'),('mapa','Mapa cromatico')]
        for i,(v,name) in enumerate(view_names):
            self.draw_button(surf,name,(39+i*172,809,164,39),lambda vv=v:self.set_view(vv),active=self.view==v,fontsize=13)
        self.draw_button(surf,'Selecionar folha / ROI',(561,809,224,39),self.roi_on,active=self.roi_edit,fontsize=13,disabled=self.frame is None)
        self.draw_button(surf,'Limpar ROI',(797,809,220,39),self.clear_roi,fontsize=13,disabled=self.roi is None)
        self.draw_adjuster(surf,42,864,'Sensibilidade da mascara','sensitivity','{:.0%}',.10,.1,.9)
        self.draw_adjuster(surf,297,864,'Opacidade','opacity','{:.0%}',.1,.1,.9)
        self.draw_adjuster(surf,554,864,'Confianca minima da IA','min_confidence','{:.0%}',.05,.2,.95)
        self.draw_button(surf,'Entender etapas  [?]',(809,874,208,37),lambda:self.toggle_help(),fontsize=14)

        panel(surf,(SIDE_X,771,SIDE_W,153),PANEL)
        label(surf,'ETAPAS DO PIPELINE',SIDE_X+16,783,14,TEXT,True)
        steps=[('01','Entrada RGB / ROI'),('02','Normalizacao ImageNet 224²'),
               ('03','CNN ONNX: 15 classes'),('04','Softmax + top 3'),
               ('05','HSV: areas amarelas/escuras')]
        for i,(code,step) in enumerate(steps):
            row_y=809+i*23
            label(surf,code,SIDE_X+18,row_y,12,GREEN,True)
            label(surf,step,SIDE_X+50,row_y,13,MUTED)

        # Barra de comandos separados da imagem: evita botoes sobre a foto.
        pygame.draw.line(surf,LINE,(22,934),(1514,934),1)

    def draw_actionbar(self,surf):
        self.draw_button(surf,'ABRIR IMAGEM  [O]',(24,116,209,29),self.choose_image,fontsize=13)
        self.draw_button(surf,'INICIAR WEBCAM  [W]' if self.source!='camera' else 'PARAR / CAPTURAR  [W]',
                         (244,116,228,29),self.toggle_camera,active=self.source=='camera',fontsize=13)
        self.draw_button(surf,'CAPTURAR  [ESPACO]',(483,116,208,29),self.freeze_frame,
                         disabled=self.source!='camera',fontsize=13)
        self.draw_button(surf,'REANALISAR',(703,116,158,29),lambda:self.schedule_inference(force=True),
                         disabled=self.frame is None,fontsize=13)
        self.draw_button(surf,'SALVAR PNG + JSON',(872,116,165,29),self.export,
                         disabled=self.frame is None,fontsize=13)
        self.draw_button(surf,'AJUDA  [?]',(SIDE_X,116,459,29),self.toggle_help,fontsize=13)

    def draw_footer(self,surf):
        status=self.worker.message if self.worker.busy else self.status
        label(surf,'STATUS',26,939,11,GREEN,True)
        label(surf,truncate(status,1150,12),85,939,12,MUTED)
        label(surf,'PYTHON + PYGAME + OPENCV + ONNX  |  IA LOCAL',1510,939,11,MUTED,False,'topright')

    def toggle_help(self):self.help_visible=not self.help_visible

    def help_panel(self,surf):
        overlay=pygame.Surface((SCREEN_W,SCREEN_H),pygame.SRCALPHA)
        overlay.fill((2,12,15,225))
        surf.blit(overlay,(0,0))
        r=panel(surf,(240,125,1056,712),(19,41,44),GREEN,17)
        label(surf,'COMO FUNCIONA O AGRO IA?',r.x+35,r.y+28,27,GREEN,True)
        label(surf,'Roteiro demonstrativo para Introducao a Inteligencia Artificial',r.x+35,r.y+70,15,MUTED)
        content=[
        ('01  Captura','Fotografia ou webcam: preferir uma folha centralizada, com boa luz e nitidez.'),
        ('02  Regiao de interesse','Arraste sobre a folha para reduzir fundo e comparar resultados com a imagem inteira.'),
        ('03  Pre-processamento','A rede recebe um recorte central RGB 224 x 224, normalizado com media/desvio ImageNet.'),
        ('04  Classificacao por IA','CNN MobileNetV3 previamente treinada; retorno de 15 logits e probabilidades softmax.'),
        ('05  Sintomas visuais (SEM IA)','Regras de cor HSV destacam pixels amarelados e escuros; NAO localizam a doenca.'),
        ('06  Analise critica','Confiança da rede NAO prova diagnostico; fotos de campo podem ser diferentes do treino.'),
        ('07  Limites','Modelo restrito a tomate, batata e pimentao; nao detecta insetos nem patogenos desconhecidos.'),
        ]
        for i,(head,body) in enumerate(content):
            y=r.y+123+i*70
            label(surf,head,r.x+38,y,16,GREEN,True)
            for j,line in enumerate(wrapped(body,900,2,14)):
                label(surf,line,r.x+39,y+26+j*19,14,MUTED)
        self.draw_button(surf,'VOLTAR AO LABORATORIO',(r.centerx-160,r.bottom-60,320,39),
                         self.toggle_help,active=True,fontsize=14)

    def draw(self):
        self.buttons=[]
        surf=self.canvas
        surf.fill(BG)
        # grade sutil no fundo
        for xx in range(0,SCREEN_W,37):pygame.draw.line(surf,(10,25,28),(xx,0),(xx,SCREEN_H),1)
        for yy in range(0,SCREEN_H,37):pygame.draw.line(surf,(10,25,28),(0,yy),(SCREEN_W,yy),1)
        self.draw_header(surf)
        self.draw_image(surf)
        self.draw_right(surf)
        self.draw_controls(surf)
        self.draw_actionbar(surf)
        self.draw_footer(surf)
        if self.help_visible:
            self.buttons=[]
            self.help_panel(surf)
        w,h=self.screen.get_size()
        scale=min(w/SCREEN_W,h/SCREEN_H)
        out_w,out_h=max(1,int(SCREEN_W*scale)),max(1,int(SCREEN_H*scale))
        image=pygame.transform.smoothscale(surf,(out_w,out_h)) if (out_w,out_h)!=(SCREEN_W,SCREEN_H) else surf
        self.screen.fill(BG)
        self.screen.blit(image,((w-out_w)//2,(h-out_h)//2))
        pygame.display.flip()

    def event_loop(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                self.running=False
            elif event.type==pygame.VIDEORESIZE:
                if event.w>=700 and event.h>=500:
                    self.screen=pygame.display.set_mode((event.w,event.h),pygame.RESIZABLE)
            elif event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:
                    if self.help_visible:self.help_visible=False
                    else:self.roi_edit=False
                elif event.key==pygame.K_o and not self.help_visible:self.choose_image()
                elif event.key==pygame.K_w and not self.help_visible:self.toggle_camera()
                elif event.key==pygame.K_SPACE and not self.help_visible:self.freeze_frame()
                elif event.key==pygame.K_s and not self.help_visible:self.export()
                elif event.key in (pygame.K_h,pygame.K_F1):self.toggle_help()
                elif event.key==pygame.K_r and not self.help_visible:self.roi_on()
                elif event.key==pygame.K_DELETE and not self.help_visible:self.clear_roi()
                elif event.key==pygame.K_1:self.view='original'
                elif event.key==pygame.K_2:self.view='evidencias'
                elif event.key==pygame.K_3:self.view='mapa'
            elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                p=self.real_to_virtual(event.pos)
                if self.help_visible:
                    for rect,fn in reversed(self.buttons):
                        if rect.collidepoint(p):fn();break
                    continue
                hit=False
                for rect,fn in reversed(self.buttons):
                    if rect.collidepoint(p):fn();hit=True;break
                if hit:continue
                if self.roi_edit and self.source=='imagem':
                    self.drag_start=self.pointer_to_image(p)
                    self.drag_now=self.drag_start
            elif event.type==pygame.MOUSEMOTION and self.drag_start:
                self.drag_now=self.pointer_to_image(self.real_to_virtual(event.pos)) or self.drag_now
            elif event.type==pygame.MOUSEBUTTONUP and event.button==1 and self.drag_start:
                end=self.pointer_to_image(self.real_to_virtual(event.pos)) or self.drag_now
                self.set_roi_drag(self.drag_start,end)
                self.drag_start=self.drag_now=None

    def loop(self):
        try:
            while self.running:
                dt=min(.08,self.clock.tick(FPS)/1000.)
                self.event_loop()
                self.update(dt)
                self.draw()
        finally:
            self.stop_camera()
            pygame.quit()


def main():
    try:AgroIALab().loop()
    except Exception:
        details=traceback.format_exc()
        try:(ROOT/'agroia_erros.log').write_text(details,encoding='utf-8')
        except Exception:pass
        print(details)
        raise

if __name__=='__main__':main()
