import csv
import numpy as np
import pandas as pd
import os.path
import re

__author__ = "JW"

""" A brief overview of the dataset:
(Source: https://www.kaggle.com/mirichoi0218/insurance/version/1#)
age			integer
sex			string (male/female)
bmi			float
children	integer
smoker		string (yes/no)
region		string

#charges	float
"""

def load_medicalcost_data(file_path):
	df = pd.read_csv(file_path)
	df = pre_processing_medicalcost(df)

def pre_processing_medicalcost(df):
	df = catogorize("sex", df)
	df = catogorize("smoker", df)
	df = catogorize("region", df)

	print(df.head())
	print("Writing to processed_medicalcost.csv...")
	df.to_csv("./dataset/MedicalCost/processed_medicalcost.csv")
	
	return df

def catogorize(column_name, df):
	df[[column_name]].fillna('NotAvailable', inplace=True)
	column_contents = df[column_name].values.tolist()
	unique_elements = set(column_contents)

	print(unique_elements)
	print(column_name + " extracted! len = " + str(len(unique_elements)))

	i = 0
	for element in unique_elements:
		i += 1
		element = str(element)
		new_column_name = column_name + "_" + element
		df.loc[(df[column_name] != element), new_column_name] = 0
		df.loc[(df[column_name] == element), new_column_name] = 1
		print(str(i) + " added.")
	df.drop([column_name], 1, inplace=True)
	print(column_name + " catogorized!")
	return df

def main():
	file_path = "./dataset/medicalcost/insurance.csv"
	load_medicalcost_data(file_path)

if __name__ == "__main__":
	main()