import os,sys
from CreditCard.Exception import CreditException
from CreditCard.logger import logging
from CreditCard.constants import *
from CreditCard.config.configuration import Configuartion
from CreditCard.util.util import read_yaml_file
from CreditCard.components import *
from CreditCard.entity.config_entity import *
from CreditCard.entity.artifact_entity import *





class DataTransformation:

    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_artifact:DataValidationArtifact    
    ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifcat = data_validation_artifact


        except Exception as e:
            raise CreditException(e,sys) from e