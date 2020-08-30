import os
import glob

# get file list
files = '/Users/songweizhi/Desktop/selected_strains/*.fna'
files = '/srv/scratch/z5039045/MetaCHIP_rebuttal/downloaded_genomes_renamed/*.fasta'
file_list = [os.path.basename(file_name) for file_name in glob.glob(files)]
print(file_list)


for each_genome in file_list:
    each_genome_name, ext = os.path.splitext(each_genome)
    os.system('/share/apps/barrnap/0.7/bin/barrnap %s.fasta > barrnap_%s.gff3' % (each_genome_name, each_genome_name))
    os.system('bedtools getfasta -fi %s.fna -bed barrnap_%s.gff3 -fo rRNA_seq_%s.fasta -name -s' % (each_genome_name, each_genome_name, each_genome_name))


