# Preparazione e formattazione dei documenti per  renderli compatibili con il parser  del doc2vec.

import json, os , re
# _________________________________________
# FUNZIONI UTILI:
def unisciPar( lista  ):
    # I contenuti degli articoli sono stati raschiati come liste di paragrafi diversi.
    # La funzione serve ad unirli in unico e trasformarlo poi nel formato desiderato invocando la funzione 'normalize'.

    articolo = ""
    for i in range (0,len(lista)):
        par= lista[i]
        articolo+=normalize(par)
    return articolo

def normalize( par):
    # Rimozione caratteri speciali, new-lines, punteggiatura e Lettere Maiuscole

    par = re.sub('[^a-zA-Z0-9 \n]', '', par)
    par = par.replace("\n", " ")
    return par.lower()
#_______________________________________
# imposto codifica utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#_______________________________________

titles=""
contents=""

base='./Dataset/json/TRUE/bbc/'
category=os.listdir(base)[1:]

# FALSE ARTICLE:
for c in category:

    filenames = os.listdir(base+c)
    for f in filenames :
        file=open(base + c+'/' + f )
        testo= file.read()
        contenuto=unisciPar(testo.split('\n')[1:])
        print contenuto
        file.close()

        # L'input del 'doc2vec' vuole che  ogni documento sia su una  riga di un file *.txt, senza punteggiatura e lowercase.
#        titolo = normalize(titolo)
        #print titolo
#        titles += titolo +'\n'
        contents += contenuto+'\n'

# Scrittuta su Txt
#out_file = open("Dataset_titoli_bbc.txt","w")
#out_file.write(titles)
#out_file.close()

out_file = open("Dataset_contenuti_bbc.txt","w")
out_file.write(contents)
out_file.close()

