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

    try:
        if caminho.endswith(".csv"):
            df = pd.read_csv(caminho)

        elif caminho.endswith(".xlsx"):
            df = pd.read_excel(caminho)
        
        else:
            logging.error("Formato de arquivo não suportado")
            raise ValueError("O arquivo deve ser .csv ou .xlsx")
        
    except FileNotFoundError:
        logging.error("Arquivo não enontrado", exc_info=True)
        raise

    logging.info(f"Dataset carregado com sucesso: {df.shape[0]} linhas x {df.shape[1]} colunas")

def executar_modelagem(PATHS: dict):
    df = ler_dados(PATHS["DATA_PATH"])
    logging.info("dados lidos")
   