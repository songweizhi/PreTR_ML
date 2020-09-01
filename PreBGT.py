import os
import glob
from Bio import SeqIO
from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def get_aa_group_pct_dict(faa_folder, aa_group_list):

    faa_file_re = '%s/*.faa' % faa_folder
    faa_file_list = [os.path.basename(file_name) for file_name in glob.glob(faa_file_re)]

    # get the pct of individual aa in all genome
    aa_pct_dict_of_dict = {}
    for faa_file in faa_file_list:

        faa_file_basename = faa_file[:-4]
        pwd_faa_file = '%s/%s' % (faa_folder, faa_file)

        # get the pct of individual aa in current genome
        aa_total_num = 0
        current_genome_aa_num_dict = {}
        for protein_record in SeqIO.parse(pwd_faa_file, 'fasta'):
            protein_seq = str(protein_record.seq)

            for aa in protein_seq:
                if aa not in current_genome_aa_num_dict:
                    current_genome_aa_num_dict[aa] = 1
                else:
                    current_genome_aa_num_dict[aa] += 1

                aa_total_num += 1

        # turn number to pct
        current_genome_aa_pct_dict = {}
        for amino_acid in current_genome_aa_num_dict:
            aa_pct = current_genome_aa_num_dict[amino_acid]/aa_total_num
            current_genome_aa_pct_dict[amino_acid] = aa_pct

        # add to dict
        aa_pct_dict_of_dict[faa_file_basename] = current_genome_aa_pct_dict


    # get the pct of aa groups
    genome_to_aa_group_total_pct_dict_of_dict = {}
    for aa_group in aa_group_list:

        genome_to_current_aa_group_total_pct_dict = {}
        for genome in aa_pct_dict_of_dict:

            current_aa_pct_dict = aa_pct_dict_of_dict[genome]

            aa_group_total_pct = 0
            for each_aa in aa_group:
                if each_aa in current_aa_pct_dict:
                    aa_group_total_pct += current_aa_pct_dict[each_aa]

            genome_to_current_aa_group_total_pct_dict[genome] = aa_group_total_pct

        genome_to_aa_group_total_pct_dict_of_dict[aa_group] = genome_to_current_aa_group_total_pct_dict

    return genome_to_aa_group_total_pct_dict_of_dict


def matrix_to_one_col(file_in, file_out, unchanged_col_num):

    df_one_col_handle = open(file_out, 'w')

    n = 0
    col_header_list = []
    for line in open(file_in):
        line_split = line.strip().split(',')
        if n == 0:
            col_header_list = line_split[unchanged_col_num:]
            df_one_col_handle.write('%s,Var,Value\n' % ','.join(line_split[:unchanged_col_num]))
        else:
            for (var, value) in zip(col_header_list, line_split[unchanged_col_num:]):
                df_one_col_handle.write('%s,%s,%s\n' % (','.join(line_split[:unchanged_col_num]), var, value))
        n += 1
    df_one_col_handle.close()


###################################################### file in/out #####################################################

notes = '''
hydrophobic amino acids: GAVLIPFMW
'''
wd = '/Users/songweizhi/Desktop/PreTR_ML'

# file and parameter in
faa_folder          = '%s/faa_files'           % wd
genome_cate_file    = '%s/Genome_category.csv' % wd
aa_group_list       = ['R', 'ED', 'RED', 'IVYWREL', 'GAVLIPFMW', 'NQ_NQED']
aa_group_list       = ['R', 'ED', 'RED', 'IVYWREL', 'NQ_NQED']
#aa_group_list       = ['R', 'ED', 'RED', 'IVYWREL', 'NQ_NQED', 'EK_QH', 'LK_Q']

prepare_traits_file = True  # True or False
plot_traits         = True  # True or False
prepare_traits_file = False  # True or False
plot_traits         = False  # True or False


validation_size         = 0.20
test_size               = 0.20
random_state            = 1
n_splits                = 10

# file out
genomic_traits_file                 = '%s/Genomic_traits.csv'                       % wd
genomic_traits_OneCol               = '%s/Genomic_traits_OneCol.csv'                % wd
genomic_traits_plot_scatter_matrix  = '%s/Genomic_traits_scatter_matrix.pdf'        % wd
genomic_traits_plot_grouped_boxplot = '%s/Genomic_traits_grouped_boxplot.pdf'       % wd
ml_plot_algorithm_comparison        = '%s/Genomic_traits_algorithm_comparison.pdf'  % wd


############################################# prepare genomic traits file ##############################################

# read in genome cate info
genome_cate_dict = {}
for each_genome in open(genome_cate_file):
    each_genome_split = each_genome.strip().split(',')
    genome_cate_dict[each_genome_split[0]] = each_genome_split[1]


# split aa group if found "_"
aa_group_list_splitted = []
for aa_group in aa_group_list:
    if '_' in aa_group:
        aa_group_split = aa_group.split('_')
        if len(aa_group_split) == 2:
            aa_group_list_splitted.append(aa_group_split[0])
            aa_group_list_splitted.append(aa_group_split[1])
        else:
            print('Found more than one "_" in amino acid group, program exited!')
            exit()
    else:
        aa_group_list_splitted.append(aa_group)


if prepare_traits_file is True:

    # get aa_group_pct_dict
    aa_group_pct_dict = get_aa_group_pct_dict(faa_folder, aa_group_list_splitted)


    # get faa_file_list
    faa_file_re = '%s/*.faa' % faa_folder
    faa_file_list = [os.path.basename(file_name) for file_name in glob.glob(faa_file_re)]


    # prepare genomic traits file
    genomic_traits_file_handle = open(genomic_traits_file, 'w')
    genomic_traits_file_handle.write('GenomeID,Category,%s\n' % ','.join(aa_group_list))

    for faa_file in sorted(faa_file_list):

        faa_file_base_name = faa_file[:-4]
        genome_cate = genome_cate_dict[faa_file_base_name]

        aa_group_pct_list = []
        for aa_group in aa_group_list:
            if '_' in aa_group:
                aa_group_numerator   = aa_group.split('_')[0]
                aa_group_denominator = aa_group.split('_')[1]
                aa_group_ratio = (aa_group_pct_dict[aa_group_numerator][faa_file_base_name])/(aa_group_pct_dict[aa_group_denominator][faa_file_base_name])
                aa_group_ratio = float("{0:.4f}".format(aa_group_ratio))
                aa_group_pct_list.append(aa_group_ratio)

            else:
                aa_group_pct_list.append(float("{0:.4f}".format(aa_group_pct_dict[aa_group][faa_file_base_name])))

        aa_group_pct_list_str = [str(i) for i in aa_group_pct_list]
        genomic_traits_file_handle.write('%s,%s,%s\n' % (faa_file_base_name, genome_cate, ','.join(aa_group_pct_list_str)))

    genomic_traits_file_handle.close()


################################################# plot genomic traits ##################################################

# read in dataframe
genomic_traits_df = read_csv(genomic_traits_file, header=0)

if plot_traits is True:

    #################### get scatter_matrix plot ####################

    genome_cate_index = []
    for i in genomic_traits_df.values[:, 1]:
        if i == 'Mesophiles':
            genome_cate_index.append(0)
        elif i == 'Psychrophiles':
            genome_cate_index.append(1)
        elif i == 'Thermophiles':
            genome_cate_index.append(2)

    genome_cate_index_array = np.array(genome_cate_index)

    scatter_matrix(genomic_traits_df, c=genome_cate_index_array, marker='o', s=6)
    plt.savefig(genomic_traits_plot_scatter_matrix)
    plt.close()
    plt.clf()

    #################### get grouped_boxplot ####################

    matrix_to_one_col(genomic_traits_file, genomic_traits_OneCol, 2)
    dataset_OneCol = read_csv(genomic_traits_OneCol, header=0)

    box_order   = ['Psychrophiles',   'Mesophiles', 'Thermophiles']
    color_order = ['dodgerblue', 'orange', 'red']
    sns.boxplot(data=dataset_OneCol, x="Var", y="Value", hue="Category", hue_order=box_order, palette=color_order)
    plt.savefig(genomic_traits_plot_grouped_boxplot)

    plt.close()
    plt.clf()

    #################### remove tmp file ####################

    os.remove(genomic_traits_OneCol)


################################################ Stats of genomic traits ###############################################

'''

# turn number list into arrary
measurement_P_arrary = np.array(measurement_P)
measurement_M_arrary = np.array(measurement_M)
measurement_T_arrary = np.array(measurement_T)

print('Psychrophiles\tmean: %s\tstd: %s' % (float("{0:.3f}".format(measurement_P_arrary.mean())), float("{0:.3f}".format(measurement_P_arrary.std()))))
print('Mesophiles\tmean: %s\tstd: %s' % (float("{0:.3f}".format(measurement_M_arrary.mean())), float("{0:.3f}".format(measurement_M_arrary.std()))))
print('Thermophiles\tmean: %s\tstd: %s' % (float("{0:.3f}".format(measurement_T_arrary.mean())), float("{0:.3f}".format(measurement_T_arrary.std()))))
print('')
print('Psychrophiles\tvs\tMesophiles\t(T-test):\tp=%s' % float("{0:.3f}".format(stats.ttest_ind(measurement_P, measurement_M, equal_var=False).pvalue)))
print('Psychrophiles\tvs\tThermophiles\t(T-test):\tp=%s' % float("{0:.3f}".format(stats.ttest_ind(measurement_P, measurement_T, equal_var=False).pvalue)))
print('Mesophiles\tvs\tThermophiles\t(T-test):\tp=%s' % float("{0:.3f}".format(stats.ttest_ind(measurement_M, measurement_T, equal_var=False).pvalue)))

'''

################################################## Compare algorithms ##################################################

# read in dataframe
genomic_traits_array = genomic_traits_df.values
genome_cate = genomic_traits_array[:, 1]
traits_matrix = genomic_traits_array[:, 2:]
traits_matrix_train, traits_matrix_validation, genome_cate_train, genome_cate_validation = train_test_split(traits_matrix, genome_cate, test_size=test_size, random_state=random_state)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
print('Compare algorithms:')
print('Algorithm\tMean\tStd')
for name, model in models:
    kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state)

    cv_results = cross_val_score(model,
                                 traits_matrix_train,
                                 genome_cate_train,
                                 cv=KFold(n_splits=n_splits, random_state=random_state),
                                 scoring='accuracy')

    results.append(cv_results)
    names.append(name)
    print('%s\t%f\t%f' % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.savefig(ml_plot_algorithm_comparison)
plt.close()
plt.clf()


################################################ Training and Prediction ###############################################

# Make predictions
algorithm_chose = LinearDiscriminantAnalysis()
algorithm_chose.fit(traits_matrix_train, genome_cate_train)
genome_cate_predictions = algorithm_chose.predict(traits_matrix_validation)

print('\nPrediction results:')
print('Algorithm chose\tLinearDiscriminantAnalysis')
print('Accuracy\t%s' % accuracy_score(genome_cate_validation, genome_cate_predictions))
print('\nClassification_report:\n%s' % classification_report(genome_cate_validation, genome_cate_predictions))


print(genome_cate_validation)
print(genome_cate_predictions)
