import os,sys
from CreditCard.Exception import CreditException
from CreditCard.logger import logging
from CreditCard.constants import *
from CreditCard.config.configuration import Configuartion
from CreditCard.util.util import *
from CreditCard.components import *
from CreditCard.entity.config_entity import *
from CreditCard.entity.artifact_entity import *
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class FeatureGenerator(BaseEstimator, TransformerMixin):
    """custom feature generator class to generate cluster class for the data
    scaler : StandardScaler clustering using kmeans++ and kneed"""

    def __init__(self, pay_x_columns , Age_column ,bil_amt_columns , pay_amt_columns,limit_bin, encoder= OneHotEncoder(sparse=False)):
        try:
            self.cluster = None
            self.pay_x = pay_x_columns
            self.age = Age_column
            self.bill_amt = bil_amt_columns
            self.pay_amt_columns = pay_amt_columns
            self.limit_bin = limit_bin
            self.encoder = encoder

        except Exception as e:
            raise CreditException(e, sys) from e


    def fit(self,X,y=None):
        data = X.copy()
        data = pd.DataFrame()
        pay_feature = lambda x: x if x < 4 else 4
        for col in self.pay_x:
            data[col] = X[col].apply(pay_feature)
        data[self.age]= pd.cut(X[self.age],[20, 25, 30, 35, 40, 50, 60, 80])
        for col in self.bill_amt:
            data[col] = pd.cut(X[col],[-350000,-1,0,25000, 75000, 200000, 2000000])
        for col in self.pay_amt_columns:
            data[col] = pd.cut(X[col],[-1, 0, 25000, 50000, 100000, 2000000])
        data[self.limit_bin] =pd.cut(X[self.limit_bin],[5000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 1100000])
        [data[col].astype("category")for col in data.columns]
        data_encoded = self.encoder.fit_transform(data)
        wcss=[]
        for i in range(1,11):
            kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
            kmeans.fit(data_encoded)
            wcss.append(kmeans.inertia_) 

        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        total_clusters=kn.knee
        logging.info(f"total cluster :{total_clusters}")
        self.cluster = KMeans(n_clusters=total_clusters, init='k-means++',random_state=42)
        self.cluster.fit(data_encoded)
        return self


    def transform(self, X, y=None):
        try:
            #self.logger.info("Transforming data")
            data = pd.DataFrame()
            pay_feature = lambda x: x if x < 4 else 4
            for col in self.pay_x:
                data[col] = X[col].apply(pay_feature)
            data[self.age]= pd.cut(X[self.age],[20, 25, 30, 35, 40, 50, 60, 80])
            for col in self.bill_amt:
                data[col] = pd.cut(X[col],[-350000,-1,0,25000, 75000, 200000, 2000000])
            for col in self.pay_amt_columns:
                data[col] = pd.cut(X[col],[-1, 0, 25000, 50000, 100000, 2000000])
            data[self.limit_bin] =pd.cut(X[self.limit_bin],[5000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 1100000])
            [data[col].astype("category")for col in data.columns]
            data_encoded = self.encoder.transform(data)
            cluster  = self.cluster.predict(data_encoded)
            generated_feature = np.c_[data_encoded , cluster]
            return generated_feature
        except Exception as e:
            raise CreditException(e, sys) from e


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

    
    def get_data_transformer_object(self)->ColumnTransformer:
        try:
            pay_x_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            bill_amt_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            Age_columns = "AGE"
            limit_columns = 'LIMIT_BAL'

            preprocessing = Pipeline(steps=[('feature_generator', FeatureGenerator(pay_amt_columns=pay_amt_columns, bil_amt_columns=bill_amt_columns,
                                                                       pay_x_columns=pay_x_columns,Age_column= Age_columns,
                                                                      limit_bin=limit_columns))])
            return preprocessing


        except Exception as e:
            raise CreditException(e,sys) from e

        #except Exception as e:
            #raise CreditException(e, sys) from e

            #num_pipeline = Pipeline(steps=[
                    #('imputer',SimpleImputer(strategy="mean")),
                    #('scaling',StandardScaler())])

           
            #logging.info(f"Numerical columns: {num_pipeline}")

           # preprocessing = ColumnTransformer(
           # transformers=[('num_pipeline',num_pipeline,numerical_columns)])
            #return num_pipeline

        #except Exception as e:
            #raise CreditException(e,sys) from e


    def over_sample_input(self , df , target_columns ):
        try :
            features = df.drop(target_columns , axis=1)
            target = df[target_columns]
            scaler = StandardScaler()
            scaler.fit(features)
            features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
            smote_model = SMOTE(sampling_strategy='minority', random_state=1965, k_neighbors=3)
            over_sampled_trainX, over_sampled_trainY = smote_model.fit_resample(X=features_scaled, y=target)
            features_over_sampled = pd.DataFrame(scaler.inverse_transform(over_sampled_trainX,) , columns=features.columns)
            return features_over_sampled , over_sampled_trainY
        except Exception as e:
            raise CreditException(e, sys) from e


    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()
            schema_file_path = self.data_validation_artifcat.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)
            target_column_name = dataset_schema[TARGET_COLUMN_KEY]
            logging.info(f"target column name")
            columns_to_cluster = dataset_schema[COLUMNS_TO_CLUSTER_KEY]
            logging.info(f"Obtaining training and test file path.")

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

           # schema_file_path = self.data_validation_artifcat.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            #train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            train_df = pd.read_csv(train_file_path , usecols=columns_to_cluster)

            test_df = pd.read_csv(test_file_path, usecols=columns_to_cluster)

            #schema = read_yaml_file(file_path=schema_file_path)

           # target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            #smote = SMOTE(random_state=11)
            #input_feature_train_df,target_feature_train_df = smote.fit_resample(input_feature_train_df,target_feature_test_df)

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[ input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")

    




