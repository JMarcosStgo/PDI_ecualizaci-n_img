import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en RGB
imagen_org = cv2.imread('1.jpg')
imagen_org_rgb = cv2.cvtColor(imagen_org, cv2.COLOR_BGR2RGB)

# Convierte la imagen a escala de grises
img_gris = cv2.cvtColor(imagen_org_rgb, cv2.COLOR_BGR2GRAY)

# Convertir a tipo de dato double
J = img_gris.astype(float)

# Definir una matriz de promedio pesado
matriz = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
matriz = matriz / np.sum(matriz)  # Normalizar la matriz para que sume 1

# Aplicar el filtro de promedio pesado
PromedioPesado = cv2.filter2D(J, -1, matriz)

# Mostrar las imágenes resultantes
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(imagen_org_rgb)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(PromedioPesado.astype(np.uint8))
plt.title('Promedio Pesado')
plt.axis('off')
plt.show()
