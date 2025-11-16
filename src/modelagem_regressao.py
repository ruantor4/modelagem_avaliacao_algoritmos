#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modelagem_regressao.py
AT2 - Modelagem e avaliação de algoritmos de regressão
para o Case de Eficiência Energética.

Objetivo:
- Ler o arquivo data/dados.csv (mesmo dataset da AT1)
- Aplicar pelo menos cinco técnicas de regressão supervisionada
- Avaliar os modelos com RMSE, MAE e R²
- Aplicar pré-processamento (normalização, padronização, seleção de features)
- Gerar um relatório em PDF com a análise comparativa entre os modelos
"""
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd

def ler_dados(caminho : str) -> pd.DataFrame:
    """
    Lê o CSV contendo o dataset e retorna um DataFrame.

    Parâmetros
    ----------
    caminho_arquivo : str
        Caminho completo até o arquivo CSV. Exemplo:
        "C:/projetos/eficiencia_energetica/data/dados.csv"

    Retorno
    -------
    df : pandas.DataFrame
        DataFrame contendo os dados lidos do CSV.
    """
    logging.info(f"lendo dados do arquivo: {caminho}")



def executar_modelagem(PATHS: dict):
    pass