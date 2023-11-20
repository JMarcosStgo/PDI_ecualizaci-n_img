import cv2
import numpy as np

# Cargar la imagen en RGB
imagen_org = cv2.imread('78.jpg')
imagen_org_rgb = cv2.cvtColor(imagen_org, cv2.COLOR_BGR2RGB)

# Convierte la imagen a escala de grises
img_gris = cv2.cvtColor(imagen_org_rgb, cv2.COLOR_BGR2GRAY)

# Aplicar el filtro gaussiano
kernel_size = (5, 5)
sigma = 0
img_gaussian = cv2.GaussianBlur(img_gris, kernel_size, sigma)

# Mostrar la imagen original y la imagen filtrada
cv2.imshow('Imagen original', imagen_org)
cv2.imshow('Imagen filtrada', img_gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()
