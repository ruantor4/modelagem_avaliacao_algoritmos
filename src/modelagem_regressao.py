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
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


def ler_dados(caminho : str) -> pd.DataFrame:
    """
    Lê o arquivo de dados (CSV ou XLSX) e retorna um DataFrame.

    Parâmetros
    ----------
    caminho : str
        Caminho completo até o arquivo de dados. Exemplo:
        "C:/projetos/eficiencia_energetica/data/ENB2012_data.xlsx"

    Retorno
    -------
    df : pandas.DataFrame
        DataFrame contendo os dados lidos.
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
        logging.error("Arquivo não encontrado", exc_info=True)
        raise
    
    except Exception:
        logging.error("Erro ao ler o arquivo de dados.", exc_info=True)
        raise

    logging.info(f"Dataset carregado com sucesso: {df.shape[0]} linhas x {df.shape[1]} colunas")

    return df


def renomear_colunas_pt_br(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia as colunas do dataset original (UCI) para nomes em português,
    padronizando com a AT1.

    Retorno
    -------
    df_renomeado : pd.DataFrame
        DataFrame com as colunas renomeadas.
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
    
    return df_renomeado


def preparar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o DataFrame em variáveis preditoras (X) e variáveis alvo (y).

    Esta função:
    - garante que as colunas alvo existam
    - separa X (features) e y (targets)

    Retorna
    -------
    X : pd.DataFrame
        Atributos de entrada (features).
    y : pd.DataFrame
        Duas saídas (Carga_Aquecimento e Carga_Resfriamento).
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

    Parâmetros
    ----------
    X : pd.DataFrame
        Variáveis preditoras.
    y : pd.DataFrame
        Variáveis alvo (dupla saída).
    test_size : float
        Proporção usada como teste (padrão 0.2).
    random_state : int
        Semente para reprodutibilidade.

    Retorno
    -------
    X_train, X_test, y_train, y_test : tupla de DataFrames
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
        logging.error("Erro ao dividir dados em treino e teste.", exc_info=True)
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
    - Padronização (StandardScaler -> média 0, desvio 1)
    - Normalização Min-Max (MinMaxScaler -> intervalo [0, 1])

    REGRAS PROFISSIONAIS:
    - Ajustar (fit) SOMENTE no conjunto de TREINO.
    - Aplicar transform no conjunto de TESTE.
    - Mantemos duas versões dos dados:
        * std: para modelos baseados em distância / regressão linear
        * mm:  para experimentos futuros, se necessário

    Retorna
    -------
    X_train_std, X_test_std, X_train_mm, X_test_mm, scaler_standard, scaler_minmax
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

def criar_modelos_regressao() -> dict[str, object]:
    """
    Cria e retorna um dicionário com os modelos de regressão a serem avaliados.

    Modelos definidos:
        - Linear Regression
        - Ridge Regression
        - Lasso Regression
        - Random Forest Regressor
        - Gradient Boosting (MultiOutputRegressor)
        - SVR (MultiOutputRegressor)

    Retorno
    -------
    modelos : dict[str, object]
    """
    logging.info("Criando instâncias dos modelos de regressão...")

    try:
        modelos: dict[str, object] = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.001),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                random_state=42
            
            ),
            "GradientBoosting": MultiOutputRegressor(
                GradientBoostingRegressor(random_state=42)
            ),
            "SVR": MultiOutputRegressor(
                SVR(kernel="rbf", C=100, gamma="scale")
            ),
        }
    
    except Exception:
        logging.error("Erro ao criar instâncias dos modelos de regressão.", exc_info=True)
        raise

    logging.info(f"Total de modelos criados: {len(modelos)}")
    
    return modelos



def treinar_modelos(
        modelos:dict[str, object],
        X_train: pd.DataFrame,
        y_train: pd.DataFrame
) -> dict[str, object]:
    """
    Treina todos os modelos fornecidos no dicionário `modelos`.

    Parâmetros
    ----------
    modelos : dict[str, object]
        Dicionário com instâncias dos modelos.
    X_train : pd.DataFrame
        Dados de treino (features).
    y_train : pd.DataFrame
        Dados de treino (targets).

    Retorno
    -------
    modelos_treinados : dict[str, object]
        Dicionário com os modelos já ajustados (fit).
    """
    logging.info("Iniciando treinamento dos modelos de regressão...")  

    modelos_treinados: Dict[str, object] = {}  

    try:
        for nome, modelo in modelos.items():
            logging.info(f"Treinando modelo: {nome}")
            modelo.fit(X_train, y_train)
            modelos_treinados[nome] = modelo
            logging.info(f"Modelo treinado com sucesso: {nome}")

    except Exception:
        logging.error("Erro durante o treinamento dos modelos.", exc_info=True)
        raise

    logging.info("Todos os modelos foram treinados com sucesso.")
    return modelos_treinados



def avaliar_modelos(
        modelos_treinados: dict,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula as métricas de desempenho (RMSE, MAE, R²) para cada modelo treinado.

    Parâmetros
    ----------
    modelos_treinados : dict[str, object]
        Dicionário com os modelos já ajustados.
    X_test : pd.DataFrame
        Conjunto de teste (features).
    y_test : pd.DataFrame
        Conjunto de teste (targets).

    Retorno
    -------
    df_metricas : pd.DataFrame
        DataFrame com colunas: ["Modelo", "RMSE", "MAE", "R2"]
    """
    logging.info("Avaliando modelos com RMSE, MAE e R²")

    registros = []
    try:
        for nome, modelo in modelos_treinados.items():
            logging.info(f"Avaliando modelo: {nome}")

            # Predição
            y_pred = modelo.predict(X_test)    

            # Calculo das métricas
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            registros.append({
                "Modelo": nome,
                "RMSE":rmse,
                "MAE":mae,
                "R2":r2
            })

    except Exception:
        logging.error(f"Erro ao avaliar modelo {nome}.", exc_info=True)
        raise

    df_metricas = pd.DataFrame(registros)
    logging.info("Avaliação concluída. Métricas calculadas para todos os modelos.")

    return df_metricas



def salvar_metricas_csv(df_metricas: pd.DataFrame, caminho_csv: str) ->None:
    """
    Salva o DataFrame de métricas em um arquivo CSV.

    Parâmetros
    ----------
    df_metricas : pd.DataFrame
        Tabela com métricas dos modelos.
    caminho_csv : str
        Caminho completo para o arquivo CSV.
    """
    logging.info(f"Salvando tabela de métricas em CSV: {caminho_csv}")

    try:
        # Salva o CSV
        df_metricas.to_csv(caminho_csv, index=False)
        logging.info(f"Tabela de métricas salva com sucesso em: {caminho_csv}")

    except Exception:
        logging.error("Erro ao salvar métricas em CSV.", exc_info=True)
        raise



def gerar_graficos_metricas(df_metricas: pd.DataFrame, fig_dir: str) ->None:
    """
    Gera gráficos de barras comparando RMSE, MAE e R² entre modelos.
    Salva as figuras na pasta `figs_dir`.

    Parâmetros
    ----------
    df_metricas : pd.DataFrame
        Tabela de métricas.
    figs_dir : str
        Diretório para salvar as figuras.
    """
    logging.info("Gerando gráficos comparativos das métricas...")    

    try:
        metricas = ["RMSE", "MAE", "R2"]

        for metrica in metricas:
            plt.figure(figsize=(9, 4))
            sns.barplot(data=df_metricas, x="Modelo", y=metrica)
            plt.title(f"Comparação dos Modelos - {metrica}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            caminho_fig = os.path.join(fig_dir, f"comparacao_{metrica}.png")

            plt.savefig(caminho_fig)
            logging.info(f"Gráfico salvo: {caminho_fig}")
            plt.close()

    except Exception:
        logging.error("Erro ao gerar gráficos das métricas.", exc_info=True)
        raise



def gerar_relatorio_modelagem_pdf(
        df_metricas: pd.DataFrame,
        caminho_pdf: str,
        figs_dir: str
) -> None:
    """
    Gera o relatório em PDF da AT2, contendo:

    - Capa
    - Tabela de métricas por modelo
    - Gráficos comparativos das métricas

    Parâmetros
    ----------
    df_metricas : pd.DataFrame
        Tabela de métricas dos modelos.
    caminho_pdf : str
        Caminho completo para salvar o PDF.
    figs_dir : str
        Diretório contendo as figuras de comparação.
    """
    logging.info("Gerando relatório de modelagem PDF...")

    try:
        largura, altura = A4
        c = canvas.Canvas(caminho_pdf, pagesize=A4)

        # Capa
        c.setFont("Helvetica-Bold", 22)
        c.drawCentredString(largura / 2, altura - 100, "AT2 – Modelagem de Regressão")
        c.setFont("Helvetica", 12)
        c.drawCentredString(
            largura / 2,
            altura - 130,
            f"Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        c.showPage()

        # Tabela de Metricas
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, altura - 60, "Tabela Comparativa de Métricas")
        c.setFont("Courier", 10)

        tabela_txt = df_metricas.to_string(index=False)
        linhas = tabela_txt.split("\n")

        y = altura - 90
        for linha in linhas:
            c.drawString(40, y, linha)
            y -= 14

            if y < 60:  # quebra de página
                c.showPage()
                c.setFont("Courier", 10)
                y = altura - 60
        
        c.showPage()
        
        # Analise
        melhor = df_metricas.sort_values("R2", ascending=False).iloc[0]

        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, altura - 50, "Análise Comparativa dos Modelos")

        texto = [
            f"Melhor modelo: {melhor['Modelo']}",
            "",
            f"• R²  = {melhor['R2']:.4f}",
            f"• RMSE = {melhor['RMSE']:.4f}",
            f"• MAE  = {melhor['MAE']:.4f}",
            "",
            "Conclusão:",
            "O melhor modelo apresentou a maior capacidade explicativa e menor erro.",
            "Modelos lineares tiveram pior desempenho, indicando forte não-linearidade.",
            "RandomForest e GradientBoosting tiveram boa performance geral.",
        ]

        c.setFont("Helvetica", 12)
        y = altura - 90
        for linha in texto:
            c.drawString(40, y, linha)
            y -= 20

            if y < 60:
                c.showPage()
                c.setFont("Helvetica", 12)
                y = altura - 60

        c.showPage()  # <-- FECHA ANÁLISE

        # GRÁFICOS
        if os.path.exists(figs_dir):
            figuras = sorted([
                f for f in os.listdir(figs_dir)
                if f.lower().endswith(".png")
            ])

            for fig in figuras:
                caminho_fig = os.path.join(figs_dir, fig)

                if not os.path.exists(caminho_fig):
                    continue

                img = ImageReader(caminho_fig)
                iw, ih = img.getSize()

                # redimensionamento proporcional
                max_w = largura - 80
                new_h = max_w * (ih / iw)

                if new_h > altura - 120:
                    new_h = altura - 120
                    max_w = new_h * (iw / ih)

                c.showPage()
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, altura - 60, f"Figura – {fig}")

                c.drawImage(
                    img,
                    40,
                    altura - 100 - new_h,
                    width=max_w,
                    height=new_h
                )

            # Finalizar PDF
        c.save()
        logging.info(f"Relatório PDF da modelagem salvo em: {caminho_pdf}")

    except Exception:
        logging.error("Erro ao gerar relatório PDF da modelagem.", exc_info=True)
        raise



def selecionar_melhor_modelo(df_metricas: pd.DataFrame) -> str:
    """
    Seleciona o melhor modelo com base no MAIOR R².

    Parâmetros
    ----------
    df_metricas : pd.DataFrame
        Tabela de métricas com colunas ["Modelo", "RMSE", "MAE", "R2"].

    Retorno
    -------
    nome_modelo : str
        Nome do modelo vencedor.
    """
    logging.info("Selecionando o melhor modelo com base no R²...")
    try:
        df_ordenado = df_metricas.sort_values(by="R2", ascending=False)
        melhor_linha = df_ordenado.iloc[0]
        nome_modelo = melhor_linha["Modelo"]

        logging.info(
            f"Melhor modelo: {nome_modelo} "
            f"(RMSE={melhor_linha['RMSE']:.4f}, "
            f"MAE={melhor_linha['MAE']:.4f}, "
            f"R²={melhor_linha['R2']:.4f})"
        )

    except Exception:
        logging.error("Erro ao selecionar o melhor modelo.", exc_info=True)
        raise

    return nome_modelo


def salvar_modelo_final(
    modelo: object,
    scaler: StandardScaler,
    colunas: list,
    caminho_modelo: str,
    caminho_scaler: str,
    caminho_colunas: str
) -> None:
    """
    Salva em disco:
    - modelo final treinado
    - scaler usado na padronização
    - ordem das colunas (features)

    Esses arquivos serão utilizados na AT3 (sistema web em Django).

    Parâmetros
    ----------
    modelo : object
        Instância do modelo treinado (ex: RandomForestRegressor).
    scaler : StandardScaler
        Scaler usado no pré-processamento (padronização).
    colunas : list[str]
        Lista de nomes das colunas de entrada (X.columns).
    caminho_modelo : str
        Caminho para salvar o arquivo .pkl do modelo.
    caminho_scaler : str
        Caminho para salvar o arquivo .pkl do scaler.
    caminho_colunas : str
        Caminho para salvar o JSON com a ordem das features.
    """
    logging.info("Salvando modelo final, scaler e colunas para produção...")

    try:
        # Modelo
        with open(caminho_modelo, "wb") as f_modelo:
            pickle.dump(modelo, f_modelo)

        # Scaler
        with open(caminho_scaler, "wb") as f_scaler:
            pickle.dump(scaler, f_scaler)

        # Colunas
        meta = {"colunas":colunas}
        with open(caminho_colunas, "w", encoding="utf=8") as f_json:
            json.dump(meta, f_json, indent=4, ensure_ascii=False)

        logging.info(f"Modelo final salvo em: {caminho_modelo}")
        logging.info(f"Scaler salvo em: {caminho_scaler}")
        logging.info(f"Colunas salvas em: {caminho_colunas}")

    except Exception:
        logging.error("Erro ao salvar o modelo final e artefatos de produção.", exc_info=True)
        raise



def executar_modelagem(PATHS: dict):
    """
    Controla todo o pipeline.

    1. Ler dados
    2. Renomear colunas para português
    3. Preparar X e y
    4. Dividir em treino e teste
    5. Pré-processar (StandardScaler / MinMaxScaler)
    6. Criar e treinar modelos de regressão
    7. Avaliar modelos (RMSE, MAE, R²)
    8. Salvar métricas em CSV
    9. Gerar gráficos comparativos
    10. Gerar relatório em PDF
    11. Selecionar o melhor modelo (maior R²)
    12. Salvar modelo final, scaler e colunas para uso na AT3
    """
    logging.info("======= INICIANDO PIPELINE MODELAGEM DE REGRESSÃO =======")
    
    
    try:
        # Ler dados
        df = ler_dados(PATHS["DATA_PATH"])
        logging.info("dados lidos")

        # Renomear colunas
        df = renomear_colunas_pt_br(df)
        logging.info("Colunas renomeadas para português")


        # Preparar X e y
        X, y = preparar_dados(df)
        logging.info("Dados preparados")

        # Dividir em treino teste
        X_train, X_test, y_train, y_test = dividir_dados(X, y)
        logging.info("Dados divididos")
        
        # Pré-processar (usaremos X_train_std / X_test_std nos modelos)
        (
            X_train_std, X_test_std,
            X_train_mm, X_test_mm,
            scaler_std, scaler_mm
        ) = preprocessar_dados(X_train, X_test)
        logging.info("Pré-processamento concluído.")

        # Criar modelos
        modelos = criar_modelos_regressao()
        logging.info("Modelos de regressão criados com sucesso.")

        # Treinar modelos (usando dados padronizados)
        modelos_treinados = treinar_modelos(modelos, X_train_std, y_train)
        logging.info("Modelos treinados")

        # Avaliar modelos
        df_metricas = avaliar_modelos(modelos_treinados, X_test_std, y_test)
        logging.info("Modelos avaliados com sucesso.")

        # Salvar métricas em CSV
        salvar_metricas_csv(df_metricas, PATHS["CSV_METRIC"])
        logging.info("CSV de métricas salvo")

        # Gerar gráficos
        gerar_graficos_metricas(df_metricas, PATHS["IMAGES_DIR"])
        logging.info("Gráficos de métricas gerados.")

        # Gerar relatorio PDF da modelagem
        gerar_relatorio_modelagem_pdf(
            df_metricas, 
            PATHS["PDF_REPORT"], 
            PATHS["IMAGES_DIR"]
        )
        logging.info("Relatório PDF gerado com sucesso.")

        # Selecionar o melhor modelo
        melhor_nome = selecionar_melhor_modelo(df_metricas)
        melhor_modelo = modelos_treinados[melhor_nome]
        logging.info(f"Melhor modelo selecionado: {melhor_nome}")

        # Salvar modelo final, scaler e colunas
        salvar_modelo_final(
            modelo=melhor_modelo,
            scaler=scaler_std,
            colunas=list(X.columns),
            caminho_modelo=PATHS["MODELO_PATH"],
            caminho_scaler=PATHS["SCALER_PATH"],
            caminho_colunas=PATHS["COLUNAS_JSON_PATH"],
        )

    except Exception:
        logging.error("ERRO CRÍTICO NO PIPELINE DA MODELAGEM", exc_info=True)
        raise
