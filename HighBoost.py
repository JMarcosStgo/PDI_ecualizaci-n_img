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

# Filtro de caja de 3x3
w = np.ones((3, 3), dtype=float)

# Convolución, copia borde de la imagen
P1ConvlR = cv2.filter2D(J, -1, w, borderType=cv2.BORDER_REPLICATE) * (1/9)

# Calcular la máscara
P2Mask = J - P1ConvlR

# Highboost
Highboost = J + 4 * P2Mask

# Desplegar varias imágenes en una sola ventana
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(imagen_org_rgb)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(Highboost, cmap='gray')
plt.title('Imagen Highboost')
plt.axis('off')

plt.show()
