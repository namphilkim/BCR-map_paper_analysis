import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from scipy.interpolate import make_interp_spline, BSpline
import random as rd

def hamming_dist(seq1,seq2):
    dist = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            dist += 1
    return dist

def get_index(file_list,column):   #Get index of column
    return file_list[0].index(column)

def extract_column(file_list,column,types):       #Extract 1D list of column as type
    column_index = get_index(file_list,column)
    column_list = []
    for i in range(1,len(file_list)):
        try:
            column_list.append(types(file_list[i][column_index]))
        except ValueError:
            column_list.append('')   
    return column_list

def get_column(file_list,index,types):
    column_list = []
    for i in range(len(file_list)):
        column_list.append(types(file_list[i][index]))
    return column_list

def set_zero(column):
    column_list = []
    min_val = min(column)
    for i in range(len(column)):
        column_list.append(column[i]-min_val)
    return column_list

def unique(seq):                                     # Return list of uniques
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def read_csv(file_name):
    file_pass = open(file_name,'r')
    list_line_pass = []
    split_line_pass = []
    for l, line in enumerate(file_pass):
        split_line_pass = line.split(',')
        list_line_pass.append(split_line_pass[:-1]+[split_line_pass[-1][:-1]])
    return list_line_pass

def read_tsv(file_name):
    file_pass = open(file_name,'r')
    list_line_pass = []
    split_line_pass = []
    for l, line in enumerate(file_pass):
        split_line_pass = line.split('\t')
        list_line_pass.append(split_line_pass[:-1]+[split_line_pass[-1][:-1]])
    return list_line_pass

def read_fasta(file_name):
    file_pass = open(file_name,'r')
    list_line_pass = []
    for l, line in enumerate(file_pass):
        list_line_pass.append(line[:-1])

    sequence_dict = {}
    temp_seq = ''
    for l,line in enumerate(list_line_pass):
        if line[0] == '>':
            sequence_dict[line[1:]] = ''
            temp_seq = line[1:]
        else:
            sequence_dict[temp_seq] += line

    return sequence_dict

def write_csv(file_list,file_name):                   # Write list into csv of file name
    with open(file_name, 'w') as f:
        for item in file_list:
            joined_item = ",".join(item)
            f.write("%s\n" % joined_item)
    return

def write_tsv(file_list,file_name):                   # Write list into csv of file name
    with open(file_name, 'w') as f:
        for item in file_list:
            joined_item = "\t".join(item)
            f.write("%s\n" % joined_item)
    return

fp = FontProperties(family="Arial", weight="bold") 
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
                "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
                "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
                "C" : TextPath((-0.366, 0), "C", size=1, prop=fp) }
COLOR_SCHEME = {'G': 'orange', 
                    'A': 'red', 
                    'C': 'blue', 
                    'T': 'darkgreen'}

PLETTERS = { "A" : TextPath((-0, 0), "A", size=1, prop=fp),
                "R" : TextPath((-0, 0), "R", size=1, prop=fp),
                "N" : TextPath((-0, 0), "N", size=1, prop=fp),
                "D" : TextPath((-0, 0), "D", size=1, prop=fp),
                "C" : TextPath((-0, 0), "C", size=1, prop=fp),
                "E" : TextPath((-0, 0), "E", size=1, prop=fp),
                "Q" : TextPath((-0, 0), "Q", size=1, prop=fp),
                "G" : TextPath((-0, 0), "G", size=1, prop=fp),
                "H" : TextPath((-0, 0), "H", size=1, prop=fp),
                "I" : TextPath((-0, 0), "I", size=1, prop=fp),
                "L" : TextPath((-0, 0), "L", size=1, prop=fp),
                "K" : TextPath((-0, 0), "K", size=1, prop=fp),
                "M" : TextPath((-0, 0), "M", size=1, prop=fp),
                "F" : TextPath((-0, 0), "F", size=1, prop=fp),
                "P" : TextPath((-0, 0), "P", size=1, prop=fp),
                "S" : TextPath((-0, 0), "S", size=1, prop=fp),
                "T" : TextPath((-0, 0), "T", size=1, prop=fp),
                "W" : TextPath((-0, 0), "W", size=1, prop=fp),
                "Y" : TextPath((-0, 0), "Y", size=1, prop=fp),
                "V" : TextPath((-0, 0), "V", size=1, prop=fp)}
PCOLOR_SCHEME = {'A': 'orange', 
                    'R': 'red', 
                    'N': 'blue', 
                    'D': 'darkgreen',
                    'C': 'yellow', 
                    'E': 'skyblue', 
                    'Q': 'purple', 
                    'G': 'black',
                    'H': 'pink', 
                    'I': 'lime', 
                    'L': 'brown', 
                    'K': 'grey',
                    'M': 'lightgreen', 
                    'F': 'darkblue', 
                    'P': 'darkred', 
                    'S': 'indigo',
                    'T': 'cyan', 
                    'W': 'magenta', 
                    'Y': 'khaki', 
                    'V': 'turquoise'}


def base_score(seq_column,count_list,length):
    num = len(seq_column)
    seq_dict = {'A':0,'C':1,'G':2,'T':3}
    A_list = list(np.zeros(length))
    C_list = list(np.zeros(length))
    G_list = list(np.zeros(length))
    T_list = list(np.zeros(length))
    base_list = [A_list,C_list,G_list,T_list]
    seq_num = 0
    for s,seq in enumerate(seq_column):
        seq_num += count_list[s]
        for i in range(min(len(seq),length)):
            base_list[seq_dict.get(seq[i])][i] += count_list[s]
    scores = []
    for x in range(length):
        scores.append([('A',base_list[0][x]/seq_num),('C',base_list[1][x]/seq_num),('G',base_list[2][x]/seq_num),('T',base_list[3][x]/seq_num)])
    
    return scores, seq_num

def letterAt(letter, x, y, yscale=1, ax=None):
    text = LETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def seq_logo(string_list,count_list,length):
    fp = FontProperties(family="Arial", weight="bold") 
    globscale = 1.35

    fig, ax = plt.subplots(figsize=(12,3))

    all_scores, num = base_score(string_list,count_list,length)
    x = 1
    maxi = 0
    for scores in all_scores:
        y = 0
        for base, score in scores:
            letterAt(base, x,y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)
    #plt.title(sample+' '+str(column)+' '+str(length)+'bp'+'('+str(num)+')')
    plt.xticks(range(1,x))
    plt.xlim((0, x)) 
    plt.ylim((0, maxi)) 
    plt.tight_layout()      
    return

def protein_base_score(seq_column,count_list,length):
    num = len(seq_column)
    seq_dict = {'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,
                'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19}
    base_list = []
    for i in range(20):
        base_list.append(list(np.zeros(length)))
    seq_num = 0
    for s,seq in enumerate(seq_column):
        if len(seq) == length:
            seq_num += count_list[s]
            for i in range(length):
                base_list[seq_dict.get(seq[i])][i] += count_list[s]
    scores = []
    for x in range(length):
        scores.append([('A',base_list[0][x]/seq_num),('R',base_list[1][x]/seq_num),('N',base_list[2][x]/seq_num),('D',base_list[3][x]/seq_num),
                      ('C',base_list[4][x]/seq_num),('E',base_list[5][x]/seq_num),('Q',base_list[6][x]/seq_num),('G',base_list[7][x]/seq_num),
                      ('H',base_list[8][x]/seq_num),('I',base_list[9][x]/seq_num),('L',base_list[10][x]/seq_num),('K',base_list[11][x]/seq_num),
                      ('M',base_list[12][x]/seq_num),('F',base_list[13][x]/seq_num),('P',base_list[14][x]/seq_num),('S',base_list[15][x]/seq_num),
                      ('T',base_list[16][x]/seq_num),('W',base_list[17][x]/seq_num),('Y',base_list[18][x]/seq_num),('V',base_list[19][x]/seq_num)])
    
    return scores, seq_num

def pletterAt(letter, x, y, yscale=1, ax=None):
    text = PLETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=PCOLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def protein_seq_logo(string_list,count_list, length):
    fp = FontProperties(family="Arial", weight="bold") 
    globscale = 1.35

    fig, ax = plt.subplots(figsize=(12,5))

    all_scores, num = protein_base_score(string_list,count_list, length)
    x = 1
    maxi = 0
    for scores in all_scores:
        y = 0
        for base, score in scores:
            pletterAt(base, x,y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)
    #plt.title(sample+' '+str(column)+' '+str(length)+'AA'+'('+str(num)+')')
    plt.xticks(range(1,x))
    plt.xlim((0, x)) 
    plt.ylim((0, maxi)) 
    plt.tight_layout()      
    return

PCHARLETTERS = { "+" : TextPath((-0, 0), "+", size=1, prop=fp),
                "N" : TextPath((-0, 0), "N", size=1, prop=fp),
                "O" : TextPath((-0, 0), "O", size=1, prop=fp),
                "H" : TextPath((-0, 0), "H", size=1, prop=fp),
                "C" : TextPath((-0, 0), "C", size=1, prop=fp),
                "G" : TextPath((-0, 0), "G", size=1, prop=fp),
                "P" : TextPath((-0, 0), "P", size=1, prop=fp)}
PCHARCOLOR_SCHEME = {'+': 'red', 
                    'N': 'blue', 
                    'O': 'skyblue', 
                    'H': 'darkgreen',
                    'C': 'yellow', 
                    'G': 'black', 
                    'P': 'darkred'}

def protein_character_base_score(seq_column,count_list,length):
    num = len(seq_column)
    seq_dict = {'+':0,'N':1,'O':2,'H':3,'C':4,'G':5,'P':6}
    base_list = []
    for i in range(8):
        base_list.append(list(np.zeros(length)))
    seq_num = 0
    for s,seq in enumerate(seq_column):
        if len(seq) == length:
            seq_num += count_list[s]
            for i in range(length):
                base_list[seq_dict.get(seq[i])][i] += count_list[s]
    scores = []
    for x in range(length):
        scores.append([('+',base_list[0][x]/seq_num),('N',base_list[1][x]/seq_num),('O',base_list[2][x]/seq_num),('H',base_list[3][x]/seq_num),
                      ('C',base_list[4][x]/seq_num),('G',base_list[5][x]/seq_num),('P',base_list[6][x]/seq_num)])
    
    return scores, seq_num

def pcharacterletterAt(letter, x, y, yscale=1, ax=None):
    text = PCHARLETTERS[letter]

    t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
        mpl.transforms.Affine2D().translate(x,y) + ax.transData
    p = PathPatch(text, lw=0, fc=PCHARCOLOR_SCHEME[letter],  transform=t)
    if ax != None:
        ax.add_artist(p)
    return p

def protein_character_seq_logo(string_list,count_list, length):
    fp = FontProperties(family="Arial", weight="bold") 
    globscale = 1.35
    
    character_dict = {'A': 'H', 
                    'R': '+', 
                    'N': 'O', 
                    'D': 'N',
                    'C': 'C', 
                    'E': 'N', 
                    'Q': 'O', 
                    'G': 'G',
                    'H': '+', 
                    'I': 'H', 
                    'L': 'H', 
                    'K': '+',
                    'M': 'H', 
                    'F': 'H', 
                    'P': 'P', 
                    'S': 'O',
                    'T': 'O', 
                    'W': 'H', 
                    'Y': 'H', 
                    'V': 'H'}

    fig, ax = plt.subplots(figsize=(12,5))
    new_string_list = []
    for string in string_list:
        new_string = ''
        for s in string:
            new_string += character_dict[s]
        new_string_list.append(new_string)

    all_scores, num = protein_character_base_score(new_string_list,count_list, length)
    x = 1
    maxi = 0
    for scores in all_scores:
        y = 0
        for base, score in scores:
            pcharacterletterAt(base, x,y, score, ax)
            y += score
        x += 1
        maxi = max(maxi, y)
    #plt.title(sample+' '+str(column)+' '+str(length)+'AA'+'('+str(num)+')')
    plt.xticks(range(1,x))
    plt.xlim((0, x)) 
    plt.ylim((0, maxi)) 
    plt.tight_layout()      
    return

def rev_comp(sequence):
    comp_dict = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    comp_seq = ''
    for i in range(len(sequence)):
        comp_seq += comp_dict.get(sequence[len(sequence)-1-i])
    return comp_seq
            
def cut_string(string,delim):
    cut_string_list = []
    indexes = []
    index = 0
    temp = 0
    for i,char in enumerate(string):
        if char == delim and temp == 0:
            cut_string_list.append(string[index:i])
            temp = 1
        elif char != delim and temp == 1:
            index = i
            temp = 0
    if string[-1] != delim:
        cut_string_list.append(string[index:])
    return cut_string_list
    
def read_blast(file_name):
    file_pass = open(file_name,'r')
    list_line_pass = []
    split_line_pass = []
    for l, line in enumerate(file_pass):
        split_line_pass = cut_string(line, ' ')
        list_line_pass.append(split_line_pass[:-1])
    return list_line_pass

def make_color_pallete(length):
    plt.rcParams["figure.figsize"] = (12,3) 
    fig,ax = plt.subplots(1,1)
    colors = []
    color_space = []
    for i in range(length):
        r = lambda: rd.randint(0,255)
        red = r()
        green = r()
        blue = r()
        for c in color_space:
            if (c[0]-red)**2 + (c[1]-green)**2 + (c[2]-blue)**2 < 300:
                red = r()
                green = r()
                blue = r()
        colors.append('#%02X%02X%02X' % (red,green,blue))
        color_space.append((red,green,blue))

    plt.bar(range(length),list(np.ones(length)),color=colors)
    for i in range(length):
        plt.text(i,0,colors[i],color='w',rotation=90,ha='center',va='bottom',font='serif',fontsize=16,fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim(-0.5,length-0.5)
    plt.ylim(0,1)
    plt.xticks([]) 
    plt.yticks([]) 
    plt.show()
    print(colors)
    return colors

def tuple_to_color(tuple_):
    r = int(255 * tuple_[0])  # IgA and IgE
    b = int(255 * tuple_[2])  # IgG
    g = int(255 * tuple_[1])  # IgM and IgD

    r_string = hex(r)[2:]  # remove the "0x" prefix
    r_string = r_string.zfill(2)
    b_string = hex(b)[2:]  # remove the "0x" prefix
    b_string = b_string.zfill(2)
    g_string = hex(g)[2:]  # remove the "0x" prefix
    g_string = g_string.zfill(2)

    color = '#' + r_string + g_string + b_string
    return color