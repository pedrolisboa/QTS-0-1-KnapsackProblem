import time
import numpy as np
import math as math
from itertools import compress
from functools import reduce
from random import randint
from math import pi,sqrt,cos,sin

N = 10 #Tamanho da vizinhanca
NumeroItens = 250
MaxWeight = 10.0
MinWeight = 1.0

#Gerar array de itens possiveis
#Pesos reais
itens = np.random.uniform(low=MinWeight,high = MaxWeight, size=(NumeroItens,))
print(itens)

#Pesos inteiros
#itens = np.random.randint(low=MinWeight,high = MaxWeight, size=(NumeroItens,))
#print(itens)

#Capacidade maxima da mochila
C = reduce(lambda x,y : x+y, itens)/2
#Vetor de ganho de cada item. O indice do item corresponde ao respectivo ganho
profits = np.vectorize(lambda x: x + 5)(itens)

#print("Espaço de itens: ",itens)
#print("Ganho dos itens", profits)
#print("Capacidade da mochila: ", C)

#As funcoes de calculo do objetivo e do peso sao iguais.
#Foram criadas duas por questao de facilidade de leitura dos passos executados

#Calculo Funcao Objetivo
def ObjFun(profits, solution):
    return reduce(lambda x,y: x+y, compress(profits, solution),0)

#Calculo peso de uma solucao
def CalcularPeso(itens,solucao):
    return reduce(lambda x,y: x+y, compress(itens, solucao),0)

#Realiza medicoes consecutivas nos qbits de forma a gerar uma solucao classica
def Medir(qindividuos):
    return np.vectorize(lambda x,y : 1 if (x > np.power(y,2)) else 0)\
                        (np.random.rand(NumeroItens), qindividuos[:,1])

#Aplica N medicoes nos q-bits para gerar as solucoes classicas
def GerarVizinhos(qindividuos, N):
    vizinhos = [np.array(Medir(qindividuos)) for i in range(N)]
    return vizinhos

#Realiza os ajustes necessários para manter as solucoes geradas dentro das restricoes de carga da mochila
def AjustarVizinhos(vizinhos,C):
    novosVizinhos = [np.array(AjustarSolucao(vizinho,C)) for vizinho in vizinhos]
    return novosVizinhos

#Realiza o reparo de uma solucao de forma a respeitar as restrições do problema
#O método utilizado é de reparo guloso Lamarckiano, i.e. remoçao consecutiva de
#itens da mochila até a satisfação das restrições
def AjustarSolucao(solucao,C):
    itensSelected = solucao.nonzero()[0]
    Peso = CalcularPeso(itens, solucao)
    while (Peso > C):
        r = np.random.randint(0,itensSelected.shape[0]-1)
        j = itensSelected[r]
        solucao[j] = 0
        Peso = Peso - itens[j]
        itensSelected = np.delete(itensSelected, r)
    return solucao

#Verifica se alguma solucao da iteração é melhor que o bestFitness até o momento
def NewBestFit(melhorSol, bestFit):
    if (ObjFun(profits,melhorSol) > ObjFun(profits,bestFit)):
        return melhorSol
    return bestFit

#Acha a melhor e a pior solucão dentro das solucoes geradas pela medição da vizinhança
def FindBestWorst(vizinhos):
    tmp = [np.array(ObjFun(profits,vizinho)) for vizinho in vizinhos]
    return (vizinhos[np.argmax(tmp)],vizinhos[np.argmin(tmp)])

#Atualiza a população de q-bits. A lista tabu é considerada dentro da função, durante o loop de aplicação das rotações.
#A aplicação da porta quânica em um q-bit k qualquer é proibido de ser aplicado(tabu) caso ambos os bit k da melhor e pior
#solução da iteração sejam, concomitantemente, 0 ou 1(Função xnor = 1 então é tabu).
def AtualizarQ(piorSol, melhorSol, qindividuos):
    theta = 0.01*pi
    
    for i in range(NumeroItens):
        modSinal = melhorSol[i] - piorSol[i]
        #Verifica se qk está no primeiro/terceiro quadrante ou segundo/quarto e modifica o sinal de theta de acordo
        if (qindividuos[i,0]*qindividuos[i,1] < 0) : modSinal *= -1
                
        Ugate = np.array([[cos(modSinal*theta), -sin(modSinal*theta)],
                        [sin(modSinal*theta),  cos(modSinal*theta)]])  #Matriz de rotação                
        #Rotação do individuo i
        qindividuos[i,:] = np.dot(Ugate,qindividuos[i,:])
    return qindividuos

qindividuos = np.zeros((NumeroItens,2))
qindividuos.fill(1/sqrt(2))
solucao = Medir(qindividuos)

bestFit = solucao

i = 0
NumIter = 1000

print("Limite da mochila: ", C)
print("Nro de iteracoes: ", NumIter)
print("Numero de itens: ", NumeroItens)
print("Peso Inicial:(Sem Reparo)", CalcularPeso(itens,bestFit))

bestFit = AjustarSolucao(bestFit,C)

print("Peso Inicial:(Com Reparo)", CalcularPeso(itens,bestFit))
print("Ganho Inicial:(Com Reparo)", CalcularPeso(profits,bestFit))

start_time = time.time()

while (i < NumIter):
    i = i + 1 
    vizinhos = GerarVizinhos(qindividuos, N) 
    vizinhos = AjustarVizinhos(vizinhos, C)
    (melhorSolucao,piorSolucao) = FindBestWorst(vizinhos)
    bestFit = NewBestFit(melhorSolucao,bestFit)
    qindividuos = AtualizarQ(melhorSolucao, piorSolucao, qindividuos)
    
print("Tempo : ", (time.time() - start_time))
print("Ganho Melhor Solucao Encontrada: ", ObjFun(profits,bestFit))
print("Peso Melhor Solucao Encontrada: ", CalcularPeso(itens,bestFit))