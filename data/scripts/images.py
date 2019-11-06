with open('../tsr_data/gt.txt', 'r') as gt:
    names = []
    seen = set()
    for line in gt.readlines():
        fname, _, _, _, _, _  = line.rstrip().split(';')
        if fname not in seen:
            seen.add(fname)
            names.append(fname)

for name in names[:150]:
    with open('../tsr_data/valid.txt', 'a+') as f:
        f.write('../tsr_data/' + name.replace('ppm', 'jpg') + '\n')

for name in names[150:]:
    with open('../tsr_data/train.txt', 'a+') as f:
        f.write('../tsr_data/' + name.replace('ppm', 'jpg') + '\n')