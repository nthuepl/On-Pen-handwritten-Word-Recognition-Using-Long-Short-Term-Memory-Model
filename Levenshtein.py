import numpy as np

default_del_map = np.array([float(1) for i in range(26)])
default_sub_map = np.array([[float(1) for i in range(26)] for j in range(26)])
default_ins_map = np.array([float(1) for i in range(26)])


def index(ch):
    return ord(ch)-ord('a')

def levenshtein(seq1, seq2, insert_costs=default_ins_map, delete_costs=default_del_map, substitute_costs=default_sub_map, len_weight=0): 

    del_map = delete_costs
    sub_map = substitute_costs
    ins_map = insert_costs

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0 for i in range(size_y)] for j in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(
                    matrix[x-1][y] + del_map[index(seq1[x-1])] ,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + ins_map[index(seq2[y-1])]
                ) + abs(x-y)*len_weight
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + del_map[index(seq1[x-1])],
                    matrix[x-1][y-1] + sub_map[index(seq1[x-1]), index(seq2[y-1])],
                    matrix[x][y-1] + ins_map[index(seq2[y-1])]
                ) + abs(x-y)*len_weight

    return (matrix[-1][-1])

def easy_levenshtein(seq1, seq2):

    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0 for i in range(size_y)] for j in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1 ,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )

    return (matrix[-1][-1])

if __name__ == '__main__':

    # au_w, bp_w, ce_w, fl_w, hn_w, rv_w = (0.15989722192473754, 0.26844730018846663, \
    #                 0.6323528324149036, 0.6527867146396427, 0.9156480941719015, 0.00916320719209418)
    # sub_map = np.array([[float(1) for i in range(26)] for j in range(26)])
    # sub_map[index('a'), index('u')] = au_w
    # sub_map[index('p'), index('b')] = bp_w
    # sub_map[index('e'), index('c')] = ce_w
    # sub_map[index('l'), index('f')] = fl_w
    # sub_map[index('r'), index('v')] = rv_w
    # sub_map[index('n'), index('h')] = hn_w

    example = 'qaicker'
    word_list = []
    tmp = []
    distance_distribution = []
    fp = open('20k.txt', 'r')
    for line in fp.readlines():
        word = line[:-1]
        tmp.append(word)
        # word_list.append(word)
        
    fp.close()
    min_d = 17
    min_p = 0
    page_index = [0]
    for i in range(1, 18):
        page_index.append(len(word_list))
        for j, word in enumerate(tmp):
            if len(word) == i:
                word_list.append(word)

    for i, word in enumerate(word_list):
        d = easy_levenshtein(example, word)
        distance_distribution.append(d)
        # distance_distribution.append(len(word))

        if d < min_d:
            min_d = d
            min_p = i

    import matplotlib.pyplot as plt
    plt.plot(distance_distribution, label='Levenshtein distance')
    plt.plot([min_p], [min_d], marker='*', markersize=5, color="red", label='target')

    print("average compare element: %s" % ( str( sum( [ float(page_index[i+2]-page_index[i]) for i in range(1, 16)] ) / 15 ) ) )
    plt.plot([page_index[len(example)], page_index[len(example)]],[0, 16], color='green', label='search area')
    plt.plot([page_index[len(example)+2], page_index[len(example)+2]],[0, 16], color='green')
    plt.yticks(np.array([i for i in range(18)]))
    plt.legend(loc='upper left')
    plt.savefig("search_in_20k_with_sort.pdf", bbox_inches='tight')
    plt.show()
