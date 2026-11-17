import unittest
import tempfile
from pathlib import Path
from unittest import mock
import numpy as np
import cv2

from analise_foliar import calculate_evidence,overlay_evidence,crop_roi,image_aspect_rect,detect_region
from modelo_foliar import LABELS, PORTUGUESE, to_tensor, softmax, crop_square_resize, LeafClassifier, label_pt


class TesteAlgoritmoVisual(unittest.TestCase):
    def test_verde_sem_manchas_em_condicoes_simples(self):
        im=np.zeros((120,150,3),np.uint8)
        im[:]=(30,165,48)
        a=calculate_evidence(im)
        self.assertLess(a.percent_marcado,1)

    def test_amarelo_produz_pixels_candidatos(self):
        im=np.zeros((120,160,3),np.uint8)
        im[:]=(228,199,31)
        a=calculate_evidence(im)
        self.assertGreater(a.percent_amarelo,40)

    def test_escurecido_pode_ser_marcado(self):
        im=np.zeros((120,160,3),np.uint8)
        im[:]=(92,50,24)
        a=calculate_evidence(im)
        self.assertGreater(a.percent_escuro,5)

    def test_evidencia_nao_muda_tamanho(self):
        image=np.zeros((200,310,3),dtype=np.uint8)
        a=calculate_evidence(image)
        for view in ['original','evidencias','mapa']:
            out=overlay_evidence(image,a,view=view)
            self.assertEqual(out.shape,image.shape)

    def test_opacidade_zero_preserva_original(self):
        im=np.zeros((96,105,3),np.uint8)
        im[:]=(210,183,37)
        a=calculate_evidence(im)
        result=overlay_evidence(im,a,opacity=0,view='original')
        np.testing.assert_array_equal(im,result)

    def test_imagem_preta_nao_explode(self):
        im=np.zeros((80,90,3),np.uint8)
        a=calculate_evidence(im)
        self.assertTrue(np.isfinite(a.percent_marcado))
        self.assertIn('Pouca iluminacao',a.qualidade)

    def test_mascara_valores_0_ou_255(self):
        im=np.random.default_rng(3).integers(0,256,(150,201,3),dtype=np.uint8)
        a=calculate_evidence(im)
        self.assertTrue(set(np.unique(a.areas_marcadas)).issubset({0,255}))

    def test_roi_recorte(self):
        im=np.arange(40*60*3,dtype=np.int32).reshape(40,60,3).astype(np.uint8)
        np.testing.assert_array_equal(crop_roi(im,(10,9,40,36)),im[9:36,10:40])

    def test_roi_invalida(self):
        with self.assertRaises(ValueError):crop_roi(np.zeros((90,100,3),np.uint8),(3,3,4,5))

    def test_fit_rect(self):
        out=image_aspect_rect(500,250,(0,0,300,300))
        self.assertEqual(out,(0,75,300,150))


class TesteClassificador(unittest.TestCase):
    def test_numero_de_classes(self):
        self.assertEqual(len(LABELS),15)
        self.assertEqual(len(PORTUGUESE),15)

    def test_nome_da_praga_e_dano_nao_inseto_detectado(self):
        self.assertIn('acaros',label_pt(LABELS[10]))

    def test_entrada_rgb_chw(self):
        im=np.zeros((400,600,3),dtype=np.uint8)
        t=to_tensor(im,224)
        self.assertEqual(t.shape,(1,3,224,224))
        self.assertEqual(t.dtype,np.float32)
        self.assertTrue(np.isfinite(t).all())

    def test_softmax_probabilidades(self):
        logits=np.array([1,3]+[0]*13)
        p=softmax(logits)
        self.assertAlmostEqual(float(p.sum()),1.,places=6)
        self.assertEqual(int(np.argmax(p)),1)

    def test_logits_invalidos_geram_erro(self):
        with self.assertRaises(ValueError):softmax(np.zeros(10))
        with self.assertRaises(ValueError):softmax(np.array([np.nan]*15))

    def test_corte_central(self):
        rgb=np.zeros((80,240,3),dtype=np.uint8)
        rgb[:,:120]=(255,0,0)
        cropped=crop_square_resize(rgb,100)
        self.assertEqual(cropped.shape,(100,100,3))

    def test_inferencia_modelo_falso_sem_download(self):
        class FakeInput:
            name='input'
            shape=[1,3,224,224]
        class FakeSession:
            def run(self,_,inputs):
                assert inputs['input'].shape==(1,3,224,224)
                output=np.zeros((1,15),dtype=np.float32)
                output[0,6]=7.5
                return [output]
        net=LeafClassifier(Path('unused'))
        net.session=FakeSession()
        net.side=224
        net.name='input'
        preds=net.predict(np.zeros((200,300,3),dtype=np.uint8))
        self.assertEqual(preds[0]['raw'],LABELS[6])
        self.assertEqual(len(preds),3)
        self.assertGreater(preds[0]['confidence'],preds[1]['confidence'])

if __name__=='__main__':unittest.main()
