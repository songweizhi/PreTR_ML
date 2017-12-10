from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from Bio.SeqRecord import SeqRecord

def export_dna_record(gene_seq, gene_id, gene_description, output_handle):
    seq_object = Seq(gene_seq, IUPAC.unambiguous_dna)
    seq_record = SeqRecord(seq_object)
    seq_record.id = gene_id
    seq_record.description = gene_description
    SeqIO.write(seq_record, output_handle, 'fasta')






output_handle = open('/Users/songweizhi/Desktop/combined_16s_uniq.fasta', 'w')
wrote_list = []
for each in SeqIO.parse('/Users/songweizhi/Desktop/combined_16s.fasta', 'fasta'):
    print(each.id)
    genome_id = each.id.split('_')[0]
    print(genome_id)
    sequence = str(each.seq)
    if genome_id not in wrote_list:
        export_dna_record(sequence, genome_id, '', output_handle)
        wrote_list.append(genome_id)

output_handle.close()
