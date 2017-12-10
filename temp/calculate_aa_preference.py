import os
import glob
import shutil
import numpy as np
from Bio import SeqIO



def turn_to_percentage(number_list):
    number_list_percent = []
    for each_element in number_list:
        each_element_percent = float("{0:.2f}".format(each_element / sum(number_list)))
        number_list_percent.append(each_element_percent)
    return number_list_percent


os.chdir('/Users/songweizhi/Desktop/input_files')

in_percent = 1

file_list = [os.path.basename(file_name) for file_name in glob.glob('/Users/songweizhi/Desktop/input_files/*.faa')]

file_list_sorted = sorted(file_list)

print(file_list_sorted)

aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

out_handle = open('/Users/songweizhi/Desktop/summary_aa.csv', 'w')
out_handle.write('%s\n' % ','.join(aa_list))
list_of_aa_count_list = []
for each_genome in file_list_sorted:
    each_genome_name, ext = os.path.splitext(each_genome)

    concatenated_seq = ''
    for each_aa in SeqIO.parse(each_genome, 'fasta'):
        each_aa_seq = str(each_aa.seq)
        #print(each_aa_seq)
        concatenated_seq += each_aa_seq

    aa_count_list = []
    for each in aa_list:
        aa_count_list.append(concatenated_seq.count(each))

    aa_count_list_in_percent = []
    if in_percent == 1:
        aa_count_list_in_percent = turn_to_percentage(aa_count_list)
        list_of_aa_count_list.append(aa_count_list_in_percent)
    else:
        list_of_aa_count_list.append(aa_count_list)

    aa_count_list_in_percent_str = []
    for each in aa_count_list_in_percent:
        aa_count_list_in_percent_str.append(str(each))


    out_handle.write('%s\n' % ','.join(aa_count_list_in_percent_str))

out_handle.close()


print(list_of_aa_count_list)


























