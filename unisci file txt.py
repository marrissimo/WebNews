filenames = ['Dataset_contenuti_bbc.txt', 'Dataset_contenuti_TRUE.txt']
with open('Dataset_contenuti_TRUE.txt', 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)