

transfer = {'A':'1000','T':'0100' ,'G':'0010','C':'0001'}

seq = 'CCCCCCCCCC'


seq_in_integer = ''
for each_base in seq:
    each_base_int = transfer[each_base]
    seq_in_integer += each_base_int

print(seq_in_integer)