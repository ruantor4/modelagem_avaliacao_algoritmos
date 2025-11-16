import os
from converter_csv import converter_xlsx_csv

def main():
    
    # Base do projeto
    BASE_PATH = os.path.abspath(os.path.dirname(__file__))
    
    # caminhos
    PATHS = {
        "BASE_PATH": BASE_PATH,
        "DATA_PATH": os.path.join(BASE_PATH, "data", "dados.csv"),
        "OUTPUT_DIR": os.path.join(BASE_PATH, "outputs"),
        "IMAGES_DIR": os.path.join(BASE_PATH, "outputs", "figs" ),
        "REPORT_MODELAGEM": os.path.join(BASE_PATH, "outputs", "relatorio_modelagem.pdf"),
        "CSV_METRIC": os.path.join(BASE_PATH, "outputs", "metricas_modelos.csv")
    }

    # Garante que as pastas existem
    os.makedirs(PATHS["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(PATHS["IMAGES_DIR"], exist_ok=True)

    converter_xlsx_csv()

if __name__ == "__main__":
    main()