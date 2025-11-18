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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    
    except Exception:
        logging.error("Erro ao ler o arquivo de dados.", exc_info=True)
        raise

    logging.info(f"Dataset carregado com sucesso: {df.shape[0]} linhas x {df.shape[1]} colunas")

    return df


def renomear_colunas_pt_br(df: pd.DataFrame) -> pd.DataFrame:
    """
        Renomeia as colunas para nomes mais amigáveis em português.
        Retorna o DataFrame com as colunas renomeadas.
    """
    logging.info("Renomeando colunas para o padrão em português...")

    map_colunas = {
        df.columns[0]: "Compacidade_Relativa",
        df.columns[1]: "Area_Superficial",
        df.columns[2]: "Area_Parede",
        df.columns[3]: "Area_Telhado",
        df.columns[4]: "Altura_Total",
        df.columns[5]: "Orientacao",
        df.columns[6]: "Area_Vidro",
        df.columns[7]: "Distribuicao_Area_Vidro",
        df.columns[8]: "Carga_Aquecimento",
        df.columns[9]: "Carga_Resfriamento",
    }
    df_renomeado = df.rename(columns=map_colunas)
    
    logging.info("Colunas renomeadas com sucesso.")
    logging.debug(f"Novos nomes das colunas: {df_renomeado.columns.tolist()}")
    
    return df_renomeado


def preparar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o DataFrame em variáveis preditoras (X) e variáveis alvo (y).

    Esta função:
    - garante que as colunas estejam renomeadas (padrão AT1)
    - separa X (features) e y (targets)
    - retorna exatamente o formato que os modelos esperam

    Retorna:
        X (pd.DataFrame): atributos de entrada
        y (pd.DataFrame): duas saídas (aquecimento e resfriamento)
    """
    logging.info("Preparando dados para modelagem...")

    try:
        # Define quais colunas sao alvos
        targets = ["Carga_Aquecimento", "Carga_Resfriamento"]

        # Verificação automatica das colunas
        for t in targets:
            if t not in df.columns:
                logging.error(f"Coluna alvo não encontrada: {t}")
                raise KeyError(f"Coluna alvo não encontrada: {t}")
        
        # X recebe todas colunas exceto targets
        X = df.drop(columns=targets)

        # y recebe somente os targets
        y = df[targets]

    except Exception:
        logging.error("Erro ao prepar dados para modelagem.", exc_info=True)
        raise

    logging.info("Dados preparados com sucesso (X e y separados).")
    logging.info(f"Formato X: {X.shape} | Formato y: {y.shape}")

    return X, y


def dividir_dados(
        X: pd.DataFrame,
        y: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide os dados em conjuntos de treino e teste.

    Parâmetros:
        X (pd.DataFrame): variáveis preditoras
        y (pd.DataFrame): variáveis alvo (dupla saída)
        test_size (float): porcentagem destinada ao conjunto de teste
        random_state (int): semente para reprodutibilidade

    Retorna:
        X_train, X_test, y_train, y_test
    """
    logging.info("Dividindo dados em treino e teste...")
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )
    
    except Exception:
        logging.error("Erro ao dividir dados.", exc_info=True)
        raise
    
    logging.info("Divisão concluída com sucesso.")
    logging.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    logging.info(f"y_train: {y_train.shape} | y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def preprocessar_dados(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, MinMaxScaler]:
    """
    Aplica pré-processamento nos dados:
    - Padronização (StandardScaler)
    - Normalização Min-Max (MinMaxScaler)

    REGRAS PROFISSIONAIS:
    ------------------------------------------
    - Ajustar (fit) SOMENTE no conjunto de TREINO.
    - O conjunto de TESTE recebe apenas transform().
    - Mantemos dois tipos de pré-processamento porque
      alguns modelos funcionam melhor com StandardScaler,
      e outros com MinMaxScaler.

    Retorna:
        X_train_std  -> padronizado
        X_test_std   -> padronizado
        X_train_mm   -> normalizado (0-1)
        X_test_mm    -> normalizado (0-1)
        scaler_standard
        scaler_minmax
    """
    logging.info("Iniciando pré-processamento dos dados...")

    try:
        # Padronização (Z-SCORE)
        scaler_standard = StandardScaler()
        X_train_std = scaler_standard.fit_transform(X_train)
        X_test_std = scaler_standard.transform(X_test)

        logging.info("Padronização concluída com sucesso (StandardScaler).")
 
        # Normalização Min-Max (0 a 1)
        scaler_minmax = MinMaxScaler()
        X_train_mm = scaler_minmax.fit_transform(X_train)
        X_test_mm = scaler_minmax.transform(X_test)

        logging.info("Normalização concluída com sucesso (MinMaxScaler).")

    except Exception:
        logging.error("Erro no pré-processamento dos dados.", exc_info=True)
        raise
    
    logging.info("Pré-processamento finalizado com sucesso.")

    return (
        X_train_std, 
        X_test_std, 
        X_train_mm, 
        X_test_mm, 
        scaler_standard, 
        scaler_minmax
    )



def executar_modelagem(PATHS: dict):
    """
    Controla todo o pipeline da AT2.
    Esta versão executa apenas as etapas concluídas:
    - ler dados
    - preparar dados
    - dividir dados
    """
    logging.info("======= INICIANDO PIPELINE MODELAGEM DE REGRESSÃO =======")
    
    
    try:
        # Ler dados
        df = ler_dados(PATHS["DATA_PATH"])
        logging.info("dados lidos")

        # 
        df = renomear_colunas_pt_br(df)
        logging.info("Colunas renomeadas para português")


        # Preparar X e y
        X, y = preparar_dados(df)
        logging.info("Dados preparados")

        # Dividir em treino teste
        X_train, X_test, y_train, y_test = dividir_dados(X, y)
        logging.info("Dados divididos")
        (
            X_train_std, X_test_std,
            X_train_mm, X_test_mm,
            scaler_std, scaler_mm
        ) = preprocessar_dados(X_train, X_test)
        logging.info("Dados Processados")


    except Exception:
        logging.error("ERRO CRÍTICO NO PIPELINE DA MODELAGEM", exc_info=True)
        raise
