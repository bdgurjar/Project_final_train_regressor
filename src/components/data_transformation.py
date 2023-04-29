import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
                    
            
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings','Restaurant_latitude', 'Restaurant_longitude','Delivery_location_longitude']
            cat_ordinal_cols=["time_for_pickup","Weather_conditions","Road_traffic_density","Vehicle_condition","multiple_deliveries","Festival","City"]
            cat_onehot_cols=["Type_of_order","Type_of_vehicle"]
            
            # Define the custom ranking for each ordinal variable
            time_pickup_cat=["5","10","15"]
            weather_cat=["Sunny","Cloudy","Fog","Windy","Stormy","Sandstorms"]
            traffic_condition_cat=["Low","Medium","High","Jam"]
            vehicle_cat=["2","1","0"]
            multiple_deliveries_cat=["0.0","1.0","2.0","3.0"]
            festival_Cat=["No","Yes"]
            city_cat=["Semi-Urban","Urban","Metropolitian"]
            vehicle_type_cat=["motorcycle","electric_scooter","scooter"]
            order_type_cat=['Snack', 'Meal', 'Drinks', 'Buffet']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )
            
            #cate ordinal
            cat_ordinal_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ("ordinalencoder",OrdinalEncoder(categories=[time_pickup_cat,weather_cat,traffic_condition_cat,vehicle_cat,multiple_deliveries_cat,festival_Cat,city_cat])),
                ("scaler",StandardScaler())
                ]
            )

            #Categorical onehote encoder pipeline
            cat_onehot_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ("onehotencoder",OneHotEncoder(categories=[order_type_cat,vehicle_type_cat],sparse_output=False)),
                ("scaler",StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ("cat_ordinal_pipeline",cat_ordinal_pipeline,cat_ordinal_cols),
            ("cat_onehot_pipeline",cat_onehot_pipeline,cat_onehot_cols)

            ])
            
            return preprocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("Error in Data Trnasformation")
            raise CustomException(e,sys)
        
    def initaite_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            
            train_df["time_for_pickup"]=train_df["time_for_pickup"].astype("str")
            train_df["Vehicle_condition"]=train_df["Vehicle_condition"].astype("str")
            train_df["multiple_deliveries"]=train_df["multiple_deliveries"].astype("str")
            
            test_df["time_for_pickup"]=test_df["time_for_pickup"].astype("str")
            test_df["Vehicle_condition"]=test_df["Vehicle_condition"].astype("str")
            test_df["multiple_deliveries"]=test_df["multiple_deliveries"].astype("str")



            target_column_name = 'Time_taken'
            drop_columns = [target_column_name,"id"]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)