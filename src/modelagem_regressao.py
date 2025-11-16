#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelagem_regressao.py
AT2 - Modelagem e avaliação de algoritmos de regressão
para o Case de Eficiência Energética.

Objetivo:
- Converter o arquivo XLSX para CSV
- Ler o arquivo data/dados.csv (mesmo dataset da AT1)
- Aplicar pelo menos cinco técnicas de regressão supervisionada
- Avaliar os modelos com RMSE, MAE e R²
- Aplicar pré-processamento (normalização, padronização, seleção de features)
- Gerar um relatório em PDF com a análise comparativa entre os modelos
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Caminhos

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_PATH, "data", "dados.csv")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "figs")
PDF_PATH = os.path.join(OUTPUT_DIR, "relatorio_modelagem.pdf")
METRICAS_CSV_PATH = os.path.join(OUTPUT_DIR, "metricas_modelos.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

