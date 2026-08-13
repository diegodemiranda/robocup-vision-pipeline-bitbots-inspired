"""
color_common.py

Módulo compartilhado pelos três detectores (field_boundary_detector.py,
obstacle_detector.py, line_detector.py). Define o espaço de cor HSV e o
ColorDetector, seguindo a Seção 3.1 "Color Detector" do artigo Bit-Bots.

>>> PONTO DE LEITURA DOS VALORES HSV REAIS <<<
A função `load_color_spaces_from_csv()` abaixo é onde você deve plugar
    pd.read_csv("meus_valores_hsv.csv")
quando tiver os valores calibrados a partir da câmera real (em vez dos
valores de exemplo/hardcoded usados para teste).
"""

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np
import pandas as pd


@dataclass
class ColorSpaceHSV:
    """Espaço de cor definido por min/max nos 3 canais HSV (OpenCV: H em 0-179)."""
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int

    def mask(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        return cv2.inRange(hsv, lower, upper)


class ColorDetector:
    """
    Classifica pixels contra espaços de cor configuráveis: campo (verde),
    marcações (branco), e marcadores de time (vermelho/azul).
    """

    def __init__(
        self,
        field_color: ColorSpaceHSV,
        marking_color: ColorSpaceHSV,
        red_color: ColorSpaceHSV,
        blue_color: ColorSpaceHSV,
        white_color: ColorSpaceHSV,
    ):
        self.field_color = field_color
        self.marking_color = marking_color
        self.red_color = red_color
        self.blue_color = blue_color
        self.white_color = white_color

    def field_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.field_color.mask(image_bgr)

    def marking_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        return self.marking_color.mask(image_bgr)


# ---------------------------------------------------------------------------
# Valores de EXEMPLO (chutes razoáveis para testes com imagem sintética).
# Substitua pelo resultado de load_color_spaces_from_csv() quando tiver
# valores calibrados de verdade.
# ---------------------------------------------------------------------------

DEFAULT_COLOR_SPACES: Dict[str, ColorSpaceHSV] = {
    "field": ColorSpaceHSV(35, 85, 50, 255, 30, 255),
    "marking": ColorSpaceHSV(0, 180, 0, 60, 180, 255),
    "red": ColorSpaceHSV(0, 10, 100, 255, 80, 255),
    "blue": ColorSpaceHSV(100, 130, 100, 255, 80, 255),
    "white": ColorSpaceHSV(0, 180, 0, 60, 180, 255),
}


def load_color_spaces_from_csv(csv_path: str = "meus_valores_hsv.csv") -> Dict[str, ColorSpaceHSV]:
    """
    >>> AQUI é onde você lê os valores HSV calibrados na câmera real. <<<

    Formato esperado do CSV (uma linha por classe de cor):
        classe,h_min,h_max,s_min,s_max,v_min,v_max
        field,35,85,50,255,30,255
        marking,0,180,0,60,180,255
        red,0,10,100,255,80,255
        blue,100,130,100,255,80,255
        white,0,180,0,60,180,255

    Ajuste os nomes de coluna/linha conforme a ferramenta que você usar para
    gerar o CSV (ex.: um script simples com trackbars do OpenCV salvando os
    valores min/max escolhidos interativamente, como sugerido no artigo).
    """
    df = pd.read_csv(csv_path)  # <-- PONTO DE LEITURA DO CSV COM VALORES REAIS

    color_spaces: Dict[str, ColorSpaceHSV] = {}
    for _, row in df.iterrows():
        color_spaces[row["classe"]] = ColorSpaceHSV(
            h_min=int(row["h_min"]), h_max=int(row["h_max"]),
            s_min=int(row["s_min"]), s_max=int(row["s_max"]),
            v_min=int(row["v_min"]), v_max=int(row["v_max"]),
        )
    return color_spaces


def build_color_detector(color_spaces: Dict[str, ColorSpaceHSV]) -> ColorDetector:
    """Monta um ColorDetector a partir de um dict de espaços de cor (do CSV ou default)."""
    return ColorDetector(
        field_color=color_spaces["field"],
        marking_color=color_spaces["marking"],
        red_color=color_spaces["red"],
        blue_color=color_spaces["blue"],
        white_color=color_spaces["white"],
    )
