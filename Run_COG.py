import os
import glob
import shutil


wd = '/Users/songweizhi/Desktop/COG_annotation'
input_files = 'input_files'
input_file_extension = 'faa'
nodes_number = 1
ppn_number = 1
memory = 10
walltime_needed = '11:59:00'
modules_needed = ['perl/5.20.1', 'python/3.5.2', 'blast+/2.6.0']
pwd_COG_wrapper = '/srv/scratch/z5039045/Scripts/COG_wrapper.py'


line_1 = '#!/bin/bash\n'
line_2 = '#PBS -l nodes=' + str(nodes_number) + ':ppn=' + str(ppn_number) + '\n'
line_3 = '#PBS -l vmem=' + str(memory) + 'gb\n'
line_4 = '#PBS -l walltime=' + walltime_needed + '\n'
line_5 = '#PBS -j oe\n'
line_7 = '#PBS -m ae\n\n'
header = line_1 + line_2 + line_3 + line_4 + line_5 + line_7
module_lines = ''
for module in modules_needed:
    module_lines += 'module load ' + module + '\n'

running_dir = 'running_directory'
qsub_folder = '0_qsub_files'

os.chdir(wd)
file_list = [os.path.basename(file_name) for file_name in glob.glob('%s/*.%s' % (input_files, input_file_extension))]

# create outputs folder
if os.path.isdir(running_dir):
    shutil.rmtree(running_dir, ignore_errors=True)
    if os.path.isdir(running_dir):
        shutil.rmtree(running_dir, ignore_errors=True)
    os.makedirs(running_dir)
    os.makedirs('%s/%s' % (running_dir, qsub_folder))
else:
    os.makedirs(running_dir)
    os.makedirs('%s/%s' % (running_dir, qsub_folder))


for genome in file_list:
    genome_name, ext = os.path.splitext(genome)

    os.mkdir('%s/%s' % (running_dir, genome_name))
    os.system('cp %s/%s %s/%s' % (input_files, genome, running_dir, genome_name))
    qsub_file_handle = open('%s/%s/qsub_COG_wrapper_%s.sh' % (running_dir, qsub_folder, genome_name), 'w')
    qsub_file_handle.write(header)
    qsub_file_handle.write(module_lines)
    qsub_file_handle.write('\ncd %s/%s/%s\n' % (wd, running_dir, genome_name))
    qsub_file_handle.write('python3 %s -in %s -t P\n' % (pwd_COG_wrapper, genome))
    qsub_file_handle.close()
    os.system('qsub %s/%s/qsub_COG_wrapper_%s.sh' % (running_dir, qsub_folder, genome_name))
