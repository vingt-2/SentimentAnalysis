'''
Vincent Petrella 
NLP Project
260467117
I hereby state that all work featured in this code was produced by myself
'''
import glob, codecs 

ENCODING = "mbcs"

def get_int_string(str):
    res = ''
    for c in str:
        if c.isdigit():
            res += c
    return int(res)

def get_datafiles(filenames):
    return glob.glob(filenames)
    
def load_file(filename):
    lines = []
    with codecs.open(filename, encoding=ENCODING) as file:
        for l in file:
            if l != '\n':
                lines.append(l)
                
    return (get_int_string(filename), lines)
    
def load_maxdiff_lexicon(folder):
    lexicon = dict()
    with open(folder+'/Maxdiff-Twitter-Lexicon_-1to1.txt') as file:
        for l in file:
            p = l.split("\t")
            lexicon[p[1].replace('#','').replace('\n','')] = float(p[0])
            
    return lexicon
    
def extract_whole_review_text(lines,author):
    text = ""
    if author == 'schwartz':
        for l in lines[2:-1]:
            text += l
    elif author == 'beradinelli':
        for l in lines[1:]:
            text += l
    elif author == 'renshaw':
        for l in lines:
            if ('On the Renshaw scale' not in l) & (':' not in l[:30]):
                text += l
    elif author == 'rhodes':
        for l in lines[:-3]:
            text += l
    else:
        print 'Invalid author name.'
        
    return lines

def load_whole_reviews():
    reviews = dict()
    
    schwartz    = get_datafiles('data/scale_whole_review/Dennis+Schwartz/txt.parag/*')  
    beradinelli = get_datafiles('data/scale_whole_review/James+Berardinelli/txt.parag/*')
    renshaw     = get_datafiles('data/scale_whole_review/Scott+Renshaw/txt.parag/*')
    rhodes      = get_datafiles('data/scale_whole_review/Steve+Rhodes/txt.parag/*')
    
    for file in schwartz:
        (id, lines) = load_file(file)
        reviews[id] = extract_whole_review_text(lines,'schwartz')
    
    for file in beradinelli:
        (id, lines) = load_file(file)
        reviews[id] = extract_whole_review_text(lines,'beradinelli')

    for file in renshaw:
        (id, lines) = load_file(file)
        reviews[id] = extract_whole_review_text(lines,'renshaw')

    for file in rhodes:
        (id, lines) = load_file(file)
        reviews[id] = extract_whole_review_text(lines,'rhodes')
    
    return reviews
    
def extract_scale_data(filenames):
    ids     = []
    labels3 = []
    labels4 = []
    rating  = []
    subj    = []
    for f in filenames:
        with codecs.open(f, encoding=ENCODING) as file:
            for l in file:
                if ('id' in f):
                    ids.append(int(l))
                if 'label.3' in f:
                    labels3.append(int(l))
                if 'label.4' in f:
                    labels4.append(int(l))
                if 'rating' in f:
                    rating.append(int(l.replace('0.','')))
                if 'subj' in f:
                    subj.append(l)
                    
    return zip(ids, labels3, labels4, rating, subj)
            
def load_scale_data():

    scale_data = dict()
    
    schwartz    = get_datafiles('data/scaledata/Dennis+Schwartz/*')  
    beradinelli = get_datafiles('data/scaledata/James+Berardinelli/*')
    renshaw     = get_datafiles('data/scaledata/Scott+Renshaw/*')
    rhodes      = get_datafiles('data/scaledata/Steve+Rhodes/*')
    
    list = extract_scale_data(schwartz) +  extract_scale_data(beradinelli) +  extract_scale_data(renshaw) +  extract_scale_data(rhodes)
    
    for (id,l3,l4,r,s) in list:
        scale_data[id] = (l3,l4,r,s)
    
    return scale_data
    
if __name__ == '__main__':
    
    #reviews     = load_whole_reviews()
    #scale_data  = load_scale_data()
    print load_maxdiff_lexicon("lexicon/Maxdiff-Twitter-Lexicon")