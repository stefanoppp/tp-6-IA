from neurona_final import Neurona_final
from neuronas_ocultas import Neurona_oculta
import matplotlib.pyplot as plt
import random

class Back_Propagation():
    
    def __init__(self,entradas,salidas):
        self.entradas=entradas
        self.salidas=salidas
    
    def main(self):
        
        cant_neuronas=int(input("Digite cantidad de neuronas: "))
        iteraciones=int(input("Digite cantidad de iteraciones: "))
        
        neuronas=[]
        
        for i in range(cant_neuronas):
            pesos_neuronales=[]
            for j in range(len(self.entradas[0])):
                peso_random=random.uniform(-0.001,0.001)
                pesos_neuronales.append(peso_random)
            n=Neurona_oculta(pesos_neuronales)
            neuronas.append(n)
            
        # generamos neurona final con 1 peso por cada neurona generada
        pesos_finales=[]
        for j in range(len(neuronas)):
            peso_random=random.random()
            pesos_finales.append(peso_random)
        nf=Neurona_final(pesos_finales)

        # ------------------comienza la iteracion------------------
        errores=[]
        # obtenemos errores neurona final y red
        for iteracion in range(iteraciones):
            for i in range(len(self.entradas)):
                salidas_ocultas=[]
                for neurona in neuronas:
                    salida_oculta=neurona.obtener_salida(self.entradas[i])
                    salidas_ocultas.append(salida_oculta)
                
                salida_red=nf.obtener_salida(salidas_ocultas)
                error_red=nf.obtener_error(self.salidas[i],salida_red)
                errores.append(error_red)
        # ---------------------obtenemos errores neuronas ocultas--------------
                errores_ocultos=[]        
                for j in range(len(neuronas)):
                    error_oculto=neuronas[j].obtener_error(nf.pesos[j],error_red,salidas_ocultas[j])
                    errores_ocultos.append(error_oculto)
                    
        # ---------------------------recalculamos pesos ocultos--------------
                for j in range(len(neuronas)):
                    neuronas[j].calcular_nuevos_pesos(self.entradas[i],errores_ocultos[j])        

        # -----------------------recalculamos pesos finales-------------------
                nf.calcular_nuevos_pesos(error_red,salidas_ocultas)
            print("Iteracion numero ",iteracion)
        array=[]
        for i in range(len(self.entradas)):
            array.append([])
        
        j=0
        for i in range(len(errores)):
            array[j].append(errores[i])
            j+=1
            if j==len(self.salidas):
                j=0
        for element in array:
            plt.plot(element)
        plt.show()