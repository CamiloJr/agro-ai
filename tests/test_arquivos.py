import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

class TesteDistribuicao(unittest.TestCase):
    def test_bat_sem_bom(self):
        raw=(ROOT/'INICIAR_WINDOWS.bat').read_bytes()
        self.assertTrue(raw.startswith(b'@echo off\r\n'))
        self.assertNotIn(b'\xef\xbb\xbf',raw)
        self.assertNotIn(b'\n',raw.replace(b'\r\n',b''))
    def test_bat_usa_venv(self):
        txt=(ROOT/'INICIAR_WINDOWS.bat').read_text(encoding='ascii')
        self.assertIn('python.exe agroia_folhas.py',txt)
    def test_sem_fotos_fabricadas(self):
        self.assertFalse(list(ROOT.glob('exemplo*')))
    def test_nao_tem_diagnostico_falso(self):
        txt=(ROOT/'agroia_folhas.py').read_text(encoding='utf-8')
        self.assertIn('nao diagnostico',txt.lower())

if __name__=='__main__':unittest.main()
