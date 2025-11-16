import os
import pandas as pd

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XLSX_PATH = os.path.join(BASE_PATH, "data", "ENB2012_data.xlsx")
CSV_PATH = os.path.join(BASE_PATH, "data", "dados.csv")

def converter_xlsx_csv():
    """
        Converte o arquivo ENB2012_data.xlsx para dados.csv.
        Executado apenas se o CSV não existir.
    """
    if os.path.exists(CSV_PATH):
        print("CSV já existe. Pulando conversão...")
        return
    
    print(f"Lendo arquivo Excel: {XLSX_PATH}")
    df = pd.read_excel(XLSX_PATH)

    print(f"Salvando CSV em: {CSV_PATH}")
    df.to_csv(CSV_PATH, index=False, encoding="utf=8")

    print("Conversão concluida com sucesso!")