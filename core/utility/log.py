import logging
import sys
from pathlib import Path
from typing import Optional

class _ColoredFormatter(logging.Formatter):
    """
    Formatter interno per aggiungere i codici colore ANSI all'output del terminale.
    Garantisce un'estetica elegante e leggibile durante l'esecuzione.
    """
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"
    
    # Formato: [Data Ora] | LIVELLO | Modulo | Messaggio
    FORMAT_STR = "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s"

    FORMATS = {
        logging.DEBUG: GREY + FORMAT_STR + RESET,
        logging.INFO: BLUE + FORMAT_STR + RESET,
        logging.WARNING: YELLOW + FORMAT_STR + RESET,
        logging.ERROR: RED + FORMAT_STR + RESET,
        logging.CRITICAL: BOLD_RED + FORMAT_STR + RESET
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT_STR)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class ExperimentLogger:
    """
    Classe Factory per generare logger standardizzati per il framework XAI.
    """
    
    @staticmethod
    def get_logger(
        name: str, 
        log_file: Optional[str] = "results/experiment.log", 
        level: int = logging.INFO
    ) -> logging.Logger:
        """
        Restituisce un logger configurato con output sia su console (a colori) che su file.
        
        Args:
            name (str): Il nome del modulo che sta loggando (es. 'TestFramework', 'XGBoost').
            log_file (str, optional): Il percorso del file di log. Di default salva in 'results/'.
            level (int): Livello di logging (default: logging.INFO).
            
        Returns:
            logging.Logger: L'istanza del logger pronta all'uso.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # Evita la duplicazione degli handler se il logger viene richiamato più volte
        if logger.hasHandlers():
            logger.handlers.clear()
            
        # 1. Console Handler (Output a colori sul terminale)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(_ColoredFormatter())
        logger.addHandler(console_handler)
        
        # 2. File Handler (Output pulito senza codici ANSI salvato su disco)
        if log_file:
            log_path = Path(log_file)
            # Crea la cartella se non esiste
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            # Stesso formato della console, ma senza colori
            file_format = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s", 
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
            
        # Impedisce la propagazione al root logger di Python (evita stampe doppie indesiderate)
        logger.propagate = False
        
        return logger