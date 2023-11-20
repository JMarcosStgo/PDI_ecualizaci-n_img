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

# Correlación
Corrl = cv2.filter2D(J, -1, w) * (1/9)

# Convolución
Convl = cv2.filter2D(J, -1, w, borderType=cv2.BORDER_CONSTANT) * (1/9)

# Convolución, copia borde de la imagen
ConvlR = cv2.filter2D(J, -1, w, borderType=cv2.BORDER_REPLICATE) * (1/9)

# Mostrar las imágenes resultantes
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(imagen_org)
plt.title('Imagen original')
plt.axis('off')


plt.subplot(1, 4, 2)
plt.imshow(Corrl.astype(np.uint8))
plt.title('Correlación')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(Convl.astype(np.uint8))
plt.title('Convolución')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(ConvlR.astype(np.uint8))
plt.title('Convolución con Replicación de Borde')
plt.axis('off')

plt.show()
