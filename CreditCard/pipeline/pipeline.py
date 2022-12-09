from CreditCard.components.data_ingestion import DataIngestion
from CreditCard.config.configuration import Configuartion
from CreditCard.constants import *
from CreditCard.entity.artifact_entity import *
from CreditCard.entity.config_entity import *
from CreditCard.Exception import CreditException
from CreditCard.logger import logging
from CreditCard.util import *
import os,sys

class Pipeline:

    def __init__(self,config: Configuartion = Configuartion()) -> None:
        try:
            self.config = config

        except Exception as e:
            raise CreditException(e,sys) from e

    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()

        except Exception as e:
            raise CreditException(e,sys) from e


    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

        except Exception as e:
            raise CreditException(e,sys) from e
    
