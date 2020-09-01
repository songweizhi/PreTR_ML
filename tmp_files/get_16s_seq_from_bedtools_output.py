
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

n = 0
output_handle = open('/Users/songweizhi/Desktop/combined_10ref.barrnap.16SrRNA.fasta', 'w')
for each_seq in SeqIO.parse('/Users/songweizhi/Desktop/combined_10ref.barrnap.rRNA.fasta', 'fasta'):
    seq_id = each_seq.id
    seq_id_new = '%s_16S_rRNA_%s' % (seq_id.split(':')[2], n)
    seq_sequence = str((each_seq.seq))
    if 2000 > len(each_seq.seq) > 1000:
        export_dna_record(seq_sequence, seq_id_new, '', output_handle)
        n += 1
output_handle.close()

