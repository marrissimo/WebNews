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

base='./Dataset/json/'
true_label=os.listdir(base+'TRUE/')[1:]
false_label=os.listdir(base+'FALSE/')

for j in range(0,len(true_label)):
    file=open(base + 'TRUE/' + true_label[j] )

    testo= file.read()[2:-2]
    jsons =testo.split(']},\n')
    end =len(testo.split(']},'))-1
    file.close()

    for i in range( 0 , end ):

        str =  jsons[i]  +']}'
        #Converto Da Json a Dict con la funzione 'loads' del modulo 'json'
        dic = json.loads(str)

        # L'input del 'doc2vec' vuole che  ogni documento sia su una  riga di un file *.txt, senza punteggiatura e lowercase.
 #       titolo = unisciPar(dic['title'][0])
 #       titles += titolo +'\n'
        contents += unisciPar(dic['content']) +'\n'

# Scrittuta su Txt
#out_file = open("Dataset_titoli_TRUE.txt","w")
#out_file.write(titles)
#out_file.close()

out_file = open("Dataset_contenuti_TRUE.txt","w")
out_file.write(contents)
out_file.close()

