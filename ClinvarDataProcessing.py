import csv
import numpy as np
import pandas as pd
import os.path
import re
from sklearn import preprocessing

__author__ = "JW"

""" A brief overview of the dataset:
(Source: https://www.kaggle.com/kevinarvai/clinvar-conflicting)
*CHROM 				Chromesome number
*POS				Position on chromosome
*AF_ESP 			Allele frequencies from GO-ESP
*AF_EXAC 			Allele frequencies from ExAC
*AF_TGP 			Allele frequencies from the 1000 genomes project
*CLNHGVS 			Top-level (primary assembly, alt, or patch) HGVS expression.
*CLNVI				The variant's clinical sources reported as tag-value pairs of database and variant identifier
*MC 				Comma separated list of molecular consequence in the form of Sequence Ontology ID|molecular_consequence
*ORIGIN				Allele origin. One or more of the following values may be added: 0 - unknown; 1 - germline; 2 - somatic; 4 - inherited; 8 - paternal; 16 - maternal; 32 - de-novo; 64 - biparental; 128 - uniparental; 256 - not-tested; 512 - tested-inconclusive; 1073741824 - other
*Consequence		Type of consequence
*IMPACT				The impact modifier for the consequence type
*SYMBOL				Gene Name
*EXON 				The exon number (out of total number)
*INTRON 			The intron number (out of total number)
*cDNA_position  	Relative position of base pair in cDNA sequence
*CDS_position 		Relative position of base pair in coding sequence
*Protein_position	Relative position of amino acid in protein
*Amino_acids 		Only given if the variant affects the protein-coding sequence
*STRAND 			Defined as + (forward) or - (reverse).
*BAM_EDIT 			Indicates success or failure of edit using BAM file
*SIFT 				The SIFT prediction and/or score, with both given as prediction(score)
*PolyPhen 			The PolyPhen prediction and/or score
*LoFtool 			Loss of Function tolerance score for loss of function variants
*CADD_PHRED 		Phred-scaled CADD score
*CADD_RAW 			Score of the deleteriousness of variants
*BLOSUM62			The majority of weak protein similaritues

#CLASS				The binary representation of the target class. 0 represents no conflicting submissions and 1 represents conflicting submissions.	
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Removed Columns
MOTIF_NAME 			The source and identifier of a transcription factor binding profile aligned at this position
MOTIF_POS 			The relative position of the variation in the aligned TFBP
HIGH_INF_POS 		A flag indicating if the variant falls in a high information position of a transcription factor binding profile (TFBP)
MOTIF_SCORE_CHANGE	The difference in motif score of the reference and variant sequences for the TFBP
Codons 				The alternative codons with the variant base in upper case
DISTANCE 			Shortest distance from variant to transcript
BIOTYPE 			Biotype of transcript or regulatory feature
SSR		 			Variant Suspect Reason Codes. One or more of the following values may be added: 0 - unspecified, 1 - Paralog, 2 - byEST, 4 - oldAlign, 8 - Para_EST, 16 - 1kg_failed, 1024 - other
Allele				The variant allele used to calculate the consequence
CLNDISDB 			Tag-value pairs of disease database name and identifier
CLNDISDBINCL 		For included Variant: Tag-value pairs of disease database name and identifier
CLNDN 				ClinVar's preferred disease name for the concept specified by disease identifiers in CLNDISDB
CLNDNINCL 			For included Variant : ClinVar's preferred disease name for the concept specified by disease identifiers in CLNDISDB
CLNSIGINCL 			Clinical significance for a haplotype or genotype that includes this variant. Reported as pairs of VariationID:clinical significance.
CLNVC 				Variant Type
REF 				Reference allele
ALT 				Alternaete allele
Feature_type	  	Type of feature. Currently one of Transcript, RegulatoryFeature, MotifFeature.
Feature 			Ensembl stable ID of feature
"""
def load_clinvar_data(file_path):
	df = pd.read_csv(file_path)
	df = pre_processing_clinvar(df)

def pre_processing_clinvar(df):
	df.drop(["REF", "ALT", "CLNDISDB", "CLNDISDBINCL", "CLNDN", "CLNDNINCL", "CLNSIGINCL", "CLNVC", "SSR", "Allele", "Feature_type",
		"BIOTYPE", "Codons", "DISTANCE", "MOTIF_NAME", "MOTIF_POS", "HIGH_INF_POS", "MOTIF_SCORE_CHANGE", "Feature"], 1, inplace=True)
	"""Catagorize chromosome number column"""
	if not os.path.isfile("./dataset/processed_1.csv"):
		df = catogorize("CHROM", df)
		df = catogorize("IMPACT", df)
		#df = catogorize("SYMBOL", df)
		df.drop(["SYMBOL"], 1, inplace=True)
		df = catogorize("BAM_EDIT", df)
		df = catogorize("SIFT", df)
		df = catogorize("PolyPhen", df)
		print(df.head())
		print("Writing to processed_1.csv...")
		df.to_csv("./dataset/processed_1.csv")
	
	df = pd.read_csv("./dataset/processed_1.csv")
	df = catogorize("CLNHGVS", df, "regex")
	df = catogorize("MC", df, ',')
	df = catogorize("Consequence", df, '&')
	df = catogorize("Amino_acids", df, '/')
	df = catogorize("CLNVI", df, ',')
	df.drop(["Unnamed: 0"], 1, inplace=True)

	"""Calculate exon and intron"""

	def get_ratio(content):
		if '/' not in str(content):
			return content 
		return float(content.split('/')[0])/float(content.split('/')[1])
	df["EXON_new"] = df["EXON"].map(lambda EXON: get_ratio(EXON))
	df["INTRON_new"] = df["INTRON"].map(lambda INTRON: get_ratio(INTRON))
	df["EXON_new"].fillna(value=0, inplace=True)
	df["INTRON_new"].fillna(value=0, inplace=True)

	print("EXON and INTRON processed!")
	print(df["INTRON_new"].head())
	df.drop(["EXON", "INTRON"], 1, inplace=True)

	"""Standardize POS, cDNA_position, CDS_position, Protein_position"""
	def standardize(column_name, df):
		df[column_name].fillna(value=0)

		def remove_range(value):
			value = str(value)
			if value.isdigit():
				return float(value)
			if '-' in value:
				if '?' in value:
					return int(''.join(re.findall("[0-9]+", value)))
				return (int(value.split('-')[0]) + int(value.split('-')[1]))/2
			if value == "nan" or value == '':
				return 0
			return value

		df[column_name] = df[column_name].apply(lambda x : remove_range(x))
		scaler = preprocessing.MinMaxScaler()
		df[column_name + "_std"]= scaler.fit_transform(df[[column_name]])
		df.drop([column_name], 1, inplace=True)
		print(column_name + " standadized!")
		return df

	tobe_unstandardized = ["POS","cDNA_position", "CDS_position", "Protein_position"]
	for col in tobe_unstandardized:
		df = standardize(col, df)

	print(df.head())
	print("Writing to processed_2.csv...")
	df.to_csv("./dataset/processed_2.csv")
	
	return df

def catogorize(column_name, df, separator=None):
	"""Convert catogorical data to different boolean columns
	Some columns have entries in multiple catogories:
	CLNHGVS: separator == "regex"
	MC: separator == ','
	CLNVI: separator == ','
	Consequence: separator == '&'
	Amino_acids: separator == '/'
	"""
	df[[column_name]].fillna(' ')
	df[column_name].replace(r'\s+', 'NotAvailable', regex=True)
	column_contents = df[column_name].values.tolist()
	if separator == "regex":
		temp = [str(''.join(re.findall("g\.[0-9]+_*[0-9]+([A-Za-z]*.*)", value))) for value in column_contents]
		column_contents = []
		for ele in temp:
			if '>' in ele:
				column_contents.append(ele)
			else:
				continue
		column_contents += ["del", "ins", "dup"]
	if separator == '/':
		"""Handling amino acids"""
		temp = column_contents
		column_contents = []
		for aa in temp:
			aa = str(aa)
			if '/' not in aa:
				column_contents += list(aa.upper())

	elif separator != None:
		temp = column_contents
		column_contents = []
		for value in temp:
			value = str(value)
			if column_name == "CLNVI": 
				if '|' in value:
					value_list = value.split('|')
					for ele in value_list:
						if ',' in ele:
							column_contents.append(ele.split(',')[0])
						elif ':' in ele:
							column_contents.append(ele.split(':')[0])
				elif ',' in value:
					column_contents.append(value.split(',')[0])
				elif ':' in value:
					column_contents.append(value.split(':')[0])
			else:
				column_contents += value.split(separator)

	unique_elements = set(column_contents)

	print(unique_elements)
	print(column_name + " extracted! len = " + str(len(unique_elements)))

	i = 0
	for element in unique_elements:
		i += 1
		element = str(element)
		new_column_name = column_name + "_" + element
		if separator == None:
			df.loc[(df[column_name] != element), new_column_name] = 0
			df.loc[(df[column_name] == element), new_column_name] = 1
		else:
			df[new_column_name] = 0
			df.loc[(df[column_name].str.contains(element, na=False, regex=False)), new_column_name] = 1
			df[new_column_name].fillna(0)
		if i%10 == 0:
			print(str(i) + " added.")
	df.drop([column_name], 1, inplace=True)
	print(column_name + " catogorized!")
	return df

def main():
	file_path = "./dataset/clinvar_conflicting.csv"
	load_clinvar_data(file_path)

if __name__ == "__main__":
	main()
