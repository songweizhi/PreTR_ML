import os
import glob
import shutil
from Bio import SeqIO
import scipy
from scipy import stats


def turn_to_percentage(number_list):
    number_list_percent = []
    for each_element in number_list:
        each_element_percent = float("{0:.2f}".format(each_element / sum(number_list)))
        number_list_percent.append(each_element_percent)
    return number_list_percent


def get_total_number(to_count, full_sequence):
    total_num = 0
    for each_element in to_count:
        total_num += full_sequence.count(each_element)
    return total_num


os.chdir('/Users/songweizhi/Desktop/input_files')

in_percent = 1

file_list = [os.path.basename(file_name) for file_name in glob.glob('/Users/songweizhi/Desktop/input_files/*.faa')]
file_list_sorted = sorted(file_list)

aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

#out_handle = open('/Users/songweizhi/Desktop/summary_aa.csv', 'w')
#out_handle.write('%s\n' % ','.join(aa_list))
list_of_aa_count_list = []

measurement_P = []
measurement_M = []
measurement_T = []

for each_genome in file_list_sorted:
    each_genome_name, ext = os.path.splitext(each_genome)

    # get concatenated aa sequence for each genome
    concatenated_seq = ''
    for each_aa in SeqIO.parse(each_genome, 'fasta'):
        each_aa_seq = str(each_aa.seq)
        concatenated_seq += each_aa_seq

    # get measurement
    P_percent = (get_total_number('P', concatenated_seq))/len(concatenated_seq)
    ED_percent = (get_total_number('ED', concatenated_seq))/len(concatenated_seq)
    RED_percent = (get_total_number('RED', concatenated_seq))/len(concatenated_seq)
    IVYWREL_percent = (get_total_number('IVYWREL', concatenated_seq))/len(concatenated_seq)
    hydrophobic_aa_percent = (get_total_number('GAVLIPFMW', concatenated_seq))/len(concatenated_seq)
    NQ_NQED_ratio = (get_total_number('NQ', concatenated_seq))/(get_total_number('NQED', concatenated_seq))

    measurement = hydrophobic_aa_percent
    measurement = float("{0:.3f}".format(measurement))

    if each_genome.startswith('P'):
        measurement_P.append(measurement)
    if each_genome.startswith('M'):
        measurement_M.append(measurement)
    if each_genome.startswith('T'):
        measurement_T.append(measurement)

    print('%s\t%s' % (each_genome_name, measurement))


print('\nTwo sample T-test between P and T:')
print(stats.ttest_ind(measurement_P, measurement_T, equal_var=False))
print('\nTwo sample T-test between P and M:')
print(stats.ttest_ind(measurement_P, measurement_M, equal_var=False))
print('\nTwo sample T-test between M and T:')
print(stats.ttest_ind(measurement_M, measurement_T, equal_var=False))

