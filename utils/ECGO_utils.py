import numpy as np
import pandas as pd
import csv

def get_label_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter=',')
    id_ec = {}
    ec_id = {}
    #skip the header
    next(csvreader,None)
    for i, rows in enumerate(csvreader):
            true_ec_lst = rows[2].split(',')
            id_ec[rows[0]] = true_ec_lst
            for ec in true_ec_lst:
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id


def get_true_labels_1(file_name):
    #modified based on CLEAN get_true_labels
    #file_name = https://ondemand.rnet.missouri.edu/pun/sys/dashboard/files/fs//cluster/pixstor/xudong-lab/yichuan/EC_data_240807/EC_test.csv
    result = open(file_name, 'r')
    csvreader = csv.reader(result, delimiter=',')
    all_label = set()
    true_label_dict = {}
    #skip the header
    next(csvreader, None)
    for row in csvreader:
        true_ec_lst = row[2].split(',')
        true_label_dict[row[0]] = true_ec_lst
        for ec in true_ec_lst:
            all_label.add(ec)
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    
    return true_label, all_label

def get_true_labels_based_annot(file_name,file_annot):
    #to do: modify this using file_annot. all label should from annotation not file_name
    #modified based on CLEAN get_true_labels
    #file_name = "/cluster/pixstor/xudong-lab/yichuan/EC_data_240807/EC_test.csv"
    #file_annot = "/cluster/pixstor/xudong-lab/yichuan/EC_data_240807/nrPDB-EC_annot.tsv"
    result = open(file_name, 'r')
    csvreader = csv.reader(result, delimiter=',')
    true_label_dict = {}
    #skip the header
    next(csvreader, None)
    for row in csvreader:
        true_ec_lst = row[2].split(',')
        true_label_dict[row[0]] = true_ec_lst
    
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    
    all_label = set()
    # get all_label from the annotation file, could have more labels than just extrasct from the test file.
    with open(file_annot, mode='r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # molecular function
        next(reader, None)  # skip the headers
        ec_numbers = {'ec': next(reader)}
        next(reader, None)  # skip the headers
        #counts = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)} #can be used later
        for row in reader:
            prot, prot_ec_numbers = row[0], row[1]
            for ec in prot_ec_numbers.split(','):
                all_label.add(ec)
            
            #will use later
            #ec_indices = [ec_numbers['ec'].index(ec_num) for ec_num in prot_ec_numbers.split(',')]
            #prot2annot[prot] = {'ec': np.zeros(len(ec_numbers['ec']), dtype=float)}
            #prot2annot[prot]['ec'][ec_indices] = 1.0
            #counts['ec'][ec_indices] += 1
    
    return true_label, all_label

#EC in total 538 classes, test in total 493 classes
#In clean they only used classes in test dataset not all dataset but in GO and EC seems use all classes in evaluation
