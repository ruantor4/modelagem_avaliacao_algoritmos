import logging
import os

from modelagem_regressao import executar_modelagem



def log_system(log_path: str):
    """
    Configura o sistema de logs da aplicação.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers={
            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        }
    )
    logging.info("======== INÍCIO DA EXECUÇÃO ========")


def main():
    
    # Base do projeto
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # caminhos
    PATHS = {
        "BASE_PATH": BASE_PATH,
        "DATA_PATH": os.path.join(BASE_PATH, "data", "ENB2012_data.xlsx"),
        "OUTPUT_DIR": os.path.join(BASE_PATH, "outputs"),
        "IMAGES_DIR": os.path.join(BASE_PATH, "outputs", "figs" ),
        "PDF_REPORT": os.path.join(BASE_PATH, "outputs", "relatorio_modelagem.pdf"),
        "CSV_METRIC": os.path.join(BASE_PATH, "outputs", "metricas_modelos.csv"),
        "LOG_PATH": os.path.join(BASE_PATH, "logs.txt")
    }

    # Garante que as pastas existem
    os.makedirs(PATHS["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(PATHS["IMAGES_DIR"], exist_ok=True)
    
    # Configura os logs
    log_system(PATHS["LOG_PATH"])

    try:
        logging.info("Executando modelagem...")
        executar_modelagem(PATHS)
        logging.info("Execução finalizada com sucesso!")
    
    except Exception as e:
        logging.error("ERRO CRÍTICO NA EXECUÇÃO", exc_info=True)


if __name__ == "__main__":
    main()