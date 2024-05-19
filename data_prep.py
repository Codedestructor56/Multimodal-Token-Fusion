import shutil
import cv2
import os
import json
import numpy as np
import pandas as pd
import re

class MedicalDatasetCreator():

    def __init__(self, dataset_name: str):
        captions_and_labels_csv_path = "data/captions_and_labels.csv"
        case_images_parquet_path = "data/case_images.parquet"
        abstracts_parquet_path = "data/abstracts.parquet"
        metadata_parquet_path = "data/metadata.parquet"
        cases_parquet_path = "data/cases.parquet"

        self.directory = os.getcwd()
        self.full_image_metadata_df = pd.read_csv(captions_and_labels_csv_path)
        self.case_images_df = pd.read_parquet(case_images_parquet_path)
        self.abstracts_df = pd.read_parquet(abstracts_parquet_path)
        self.full_metadata = pd.read_parquet(metadata_parquet_path)
        self.full_cases = pd.read_parquet(cases_parquet_path)
        self.dataset_name = dataset_name
        
        self.loaded_df = None
        self.image = None
        self.case_text = None
        self.gender = None
        self.age = None
        self.image_labels = None
        self.dataset_length = None
        
        self.questions = ["What is going on in the image?","Describe a case?", "What is going on here?"]


    def create_dataset(self, im_dir: str):
                
        new_df = self.full_image_metadata_df.copy()
        pattern = r'[^a-zA-Z0-9\s]'
        new_df['article_id'] = new_df['file'].apply(lambda x: re.sub(pattern, "", x[:11]))
        
        merged_df = pd.merge(new_df[['file', 'article_id']],self.full_cases, on = "article_id",
                             how='left')
        image_files = []
        for root, dirs, files in os.walk(im_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_files.append(file)
        
        merged_df = merged_df[merged_df['file'].isin(image_files)]
        merged_df.reset_index(drop = True, inplace = True)
        unrolled_merge = pd.json_normalize(merged_df['cases'].explode())
        df_concatenated = pd.concat([merged_df.drop(columns=['cases']), unrolled_merge], axis=1)
        del merged_df, unrolled_merge
        if os.path.exists(self.dataset_name):
            raise "Dataset already exists, just load it."
        else:
            os.makedirs(f"{self.directory}/{self.dataset_name}")
    
        for root, dirs, files in os.walk(im_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    shutil.copy(os.path.join(root, file), f"{self.directory}/{self.dataset_name}")

        
        file_name = 'info_dataframe.csv'
        file_path = os.path.join(self.dataset_name,file_name)
        df_concatenated.to_csv(file_path, index=False)
        del df_concatenated, image_files
        print("Dataset created!")

    def load_dataset(self):
        self.loaded_df = pd.read_csv(os.path.join(self.dataset_name, "info_dataframe.csv"))
        self.loaded_df.dropna(inplace = True)
        self.dataset_length = self.loaded_df.shape[0]

    def load_row(self, index: int = 0):
        self.image = cv2.imread(os.path.join(self.dataset_name,self.loaded_df["file"].iloc[index]))
        self.case_text = f"{self.loaded_df["case_text"].iloc[index]}"
        self.gender = self.loaded_df["gender"].iloc[index]
        self.age = self.loaded_df["age"].iloc[index]
       # self.image_labels = self.loaded_df[""]

if __name__ == "__main__":
    md = MedicalDatasetCreator("ct_scan_data")
    md.load_dataset()
    md.load_row(100)
    print(md.case_text)
    print(md.image.shape)
    cv2.imshow("pat_im",md.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
