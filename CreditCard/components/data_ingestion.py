from CreditCard.entity.config_entity import *
from CreditCard.Exception import CreditException
import sys,os
from CreditCard.logger import logging


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20}")
            self.data_ingestion_config =data_ingestion_config

        except Exception as e:
            raise CreditException(e,sys) from e

def initiate_data_ingestion(self) ->DataIngestionArtifact:
    try:
        pass
    except Exception as e:
        raise CreditException (e,sys) from e