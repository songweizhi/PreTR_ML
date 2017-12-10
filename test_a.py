


lll = 'ABHFBCBDJFITMEYKSLDOVPDMFHBSJENFJD'
count = 'AE'


def get_total_number(to_count, full_sequence):
    total_num = 0
    for each_element in to_count:
        total_num += full_sequence.count(each_element)
    return total_num




print(get_total_number(count, lll))