from CreditCard.pipeline.pipeline import Pipeline
from CreditCard.Exception import CreditException
from CreditCard.logger import logging
from CreditCard.config.configuration import Configuartion
from CreditCard.components.data_transformation import DataTransformation
import os,sys

def main():
    try:
        #config_path = os.path.join("config","config.yaml")
        #pipeline = Pipeline(Configuartion(config_file_path=config_path))
        pipeline = Pipeline()
        pipeline.run_pipeline()
        #pipeline.start()
        #logging.info("main function execution completed.")
        # # data_validation_config = Configuartion().get_data_transformation_config()
        # # print(data_validation_config)
        # schema_file_path=r"D:\Project\machine_learning_project\config\schema.yaml"
        # file_path=r"D:\Project\machine_learning_project\housing\artifact\data_ingestion\2022-06-27-19-13-17\ingested_data\train\housing.csv"

        # df= DataTransformation.load_data(file_path=file_path,schema_file_path=schema_file_path)
        # print(df.columns)
        # print(df.dtypes)
        #data_validation_config= Configuartion().get_data_validation_config()
        #print(data_validation_config)

        #data_transformation_config= Configuartion().get_data_transformation_config()
        #print(data_transformation_config)

        #schema_file_path = r"C:\project\Credit card prediction\config\schema.yaml"
        #file_path = r"C:\project\Credit card prediction\CreditCard\artifact\data_ingestion\2022-12-20-20-31-13\ingested_data\train\UCI_Credit_Card.csv"

        #df = DataTransformation.load_data(file_path=file_path,schema_file_path=schema_file_path)
        #print(df.columns)
        #print(df.dtypes)
          
        #model_trainer_config = Configuartion().get_model_trainer_config()
        #print(model_trainer_config)

    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()