import cv2
from back_propagation import Back_Propagation

def main():
  fotos=['1A58099.jpg','1B58099.jpg','2A58099.jpg','2B58099.jpg','3A58099.jpg','3B58099.jpg','4A58099.jpg','4B58099.jpg','5A58099.jpg','5B58099.jpg']
  
  salidas=[1,0,1,0,1,0,1,0,1,0]
  
  pixeles_fotos=[]
  for foto in fotos:
    intermedio=[]
    image = cv2.imread(foto)
    for pixel in image:
      for value in pixel:
        intermedio.append(value[0])
    pixeles_fotos.append(intermedio)
  
  back=Back_Propagation(pixeles_fotos,salidas)
  back.main()
main()