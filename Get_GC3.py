import os
import glob
import statistics
import numpy as np
from Bio import SeqIO
from scipy import stats
import statistics
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



def sep_path_basename_ext(file_in):

    # separate path and file name
    file_path, file_name = os.path.split(file_in)
    if file_path == '':
        file_path = '.'

    # separate file basename and extension
    file_basename, file_extension = os.path.splitext(file_name)

    return file_path, file_basename, file_extension


def split_sequence(original_seq):
    n = 0
    original_seq_split = []
    while n+2 < len(original_seq):
        original_seq_split.append(original_seq[n: n+3])
        n += 3
    return original_seq_split


def get_GC3_value(ffn_file):

    GC3_str = ''
    for orf in SeqIO.parse(ffn_file, 'fasta'):
        orf_seq_split = split_sequence(str(orf.seq))
        current_orf_GC3_str = ''
        for codon in orf_seq_split:
            current_orf_GC3_str += codon[2]
        GC3_str += current_orf_GC3_str

    # get GC3_value
    GC3_value = float("{0:.3f}".format((GC3_str.count('G') + GC3_str.count('C') + GC3_str.count('g') + GC3_str.count('c'))*100/len(GC3_str)))

    return GC3_value


def T_test(num_list_1, num_list_2):

    # turn list to arrary
    num_list_1_arrary = np.array(num_list_1)
    num_list_2_arrary = np.array(num_list_2)

    # get mean and stdev
    num_list_1_mean = statistics.mean(num_list_1_arrary)
    num_list_2_mean = statistics.mean(num_list_2_arrary)
    num_list_1_stdev = statistics.stdev(num_list_1_arrary)
    num_list_2_stdev = statistics.stdev(num_list_2_arrary)

    # perform t_test
    t_test= stats.ttest_ind(num_list_1_arrary,num_list_2_arrary)

    # turn num list to str list
    num_list_1_str = [str(i) for i in num_list_1]
    num_list_2_str = [str(i) for i in num_list_2]

    # report
    print('Num list 1: %s' % ','.join(num_list_1_str))
    print('Num list 2: %s' % ','.join(num_list_2_str))
    print('Num list 1\tmean:%s\tstdev:%s' % (float("{0:.2f}".format(num_list_1_mean)), float("{0:.2f}".format(num_list_1_stdev))))
    print('Num list 2\tmean:%s\tstdev:%s' % (float("{0:.2f}".format(num_list_2_mean)), float("{0:.2f}".format(num_list_2_stdev))))
    print('P-value: %s' % float("{0:.3f}".format(t_test.pvalue)))


##################################################### Liu Qing CR ######################################################

# ffn_file_re = '/Users/songweizhi/Desktop/GC3/LQ_78_Prodigal_prodigal_ffn/*.ffn'
# ffn_file_list = [os.path.basename(file_name) for file_name in glob.glob(ffn_file_re)]
#
#
# CR_to_T_dict = {}
# for strain in open('/Users/songweizhi/Desktop/CT_to_T.txt'):
#     strain_split = strain.strip().split(',')
#     CR_to_T_dict[strain_split[0]] = float(strain_split[1])
#
#
# high_T_GC3_list = []
# low_T_GC3_list = []
# for ffn_file in sorted(ffn_file_list):
#
#     pwd_ffn_file = '/Users/songweizhi/Desktop/GC3/LQ_78_Prodigal_prodigal_ffn/%s' % ffn_file
#     ffn_file_path, ffn_file_basename, ffn_file_ext = sep_path_basename_ext(pwd_ffn_file)
#     current_ffn_file_T = CR_to_T_dict[ffn_file_basename]
#     current_ffn_GC3 = get_GC3_value(pwd_ffn_file)
#
#     if current_ffn_file_T <= 20:
#         low_T_GC3_list.append(current_ffn_GC3)
#     else:
#         high_T_GC3_list.append(current_ffn_GC3)
#
#
# # get box plot
# MAG_HGT_num_lol_arrary = [np.array(low_T_GC3_list), np.array(high_T_GC3_list)]
# fig = plt.figure(1, figsize=(9, 6))
# ax = fig.add_subplot(111)
# bp = ax.boxplot(MAG_HGT_num_lol_arrary)
# ax.set_xticklabels(['<=20', '>20'], rotation=0, fontsize=12)
#
# ## change the style of fliers and their fill
# for flier in bp['fliers']:
#     flier.set(marker='+', color='black', alpha=0.7, markersize=3)
#
# plt.tight_layout()
# fig.savefig('/Users/songweizhi/Desktop/CT_to_T.png', bbox_inches='tight', dpi=300)
# plt.close()
#
#
# T_test(high_T_GC3_list, low_T_GC3_list)


############################################## selected M P and T genomes ##############################################

ffn_file_folder = '/Users/songweizhi/Desktop/PMT_GC3_ffn_files'


ffn_file_re = '%s/*.ffn' % ffn_file_folder
ffn_file_list = [os.path.basename(file_name) for file_name in glob.glob(ffn_file_re)]


M_GC3_list = []
P_GC3_list = []
T_GC3_list = []
for ffn_file in sorted(ffn_file_list):

    pwd_ffn_file = '%s/%s' % (ffn_file_folder, ffn_file)
    ffn_file_path, ffn_file_basename, ffn_file_ext = sep_path_basename_ext(pwd_ffn_file)
    current_ffn_GC3 = get_GC3_value(pwd_ffn_file)

    if ffn_file_basename.startswith('M'):
        M_GC3_list.append(current_ffn_GC3)

    if ffn_file_basename.startswith('P'):
        P_GC3_list.append(current_ffn_GC3)

    if ffn_file_basename.startswith('T'):
        T_GC3_list.append(current_ffn_GC3)

    print('%s\t%s' % (ffn_file_basename, current_ffn_GC3))



# get box plot
MAG_HGT_num_lol_arrary = [np.array(P_GC3_list), np.array(M_GC3_list), np.array(T_GC3_list)]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(MAG_HGT_num_lol_arrary)
ax.set_xticklabels(['Psychrophiles', 'Mesophile', 'Thermophile'], rotation=0, fontsize=12)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='+', color='black', alpha=0.7, markersize=3)

plt.tight_layout()
fig.savefig('/Users/songweizhi/Desktop/PMT_GC3.png', bbox_inches='tight', dpi=300)
plt.close()


T_test(M_GC3_list, T_GC3_list)
print()
T_test(M_GC3_list, P_GC3_list)
print()
T_test(P_GC3_list, T_GC3_list)
