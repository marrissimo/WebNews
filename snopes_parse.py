# Preparazione e formattazione dei documenti per  renderli compatibili con il parser  del doc2vec.

import  re
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
    #('utf-8')
#_______________________________________

titles=""
contents=""



file = open("Dataset_contenuti_bbc.txt","r")
text= file.read()
articles = text.split('\n') [ :50]
output=""
for a in articles:
    output+= a+"\n"


file.close()


#file2 = open("temp.txt","r")
#nuovo= file2.read()

#file2.close()



#nuovo = unisciPar( nuovo.split('\n'))
#text+='\n'+nuovo
#text_o = ""
#for t in text.split('\n'):
#    text_o += normalize( str(t))+'\n'


# L'input del 'doc2vec' vuole che  ogni documento sia su una  riga di un file *.txt, senza punteggiatura e lowercase.
#titolo = normalize(titolo)
#print titolo
#titles += titolo +'\n'
#contents += contenuto+'\n'



out_file = open("TEST_TRUE_2_bbc_50.txt","w")
out_file.write( output)
out_file.close()

