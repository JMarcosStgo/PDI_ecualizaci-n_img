import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en RGB
imagen_org = cv2.imread('2.jpg')
imagen_org_rgb = cv2.cvtColor(imagen_org, cv2.COLOR_BGR2RGB)

# Convierte la imagen a escala de grises
img_gris = cv2.cvtColor(imagen_org_rgb, cv2.COLOR_BGR2GRAY)

# Convertir a tipo de dato double
J = img_gris.astype(float)

# Filtros Laplacianos
L1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
L2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
L3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# Aplicar correlación con los filtros Laplacianos
corrl1 = cv2.filter2D(J, -1, L1)
corrl2 = cv2.filter2D(J, -1, L2)
corrl3 = cv2.filter2D(J, -1, L3)

# Aplicar los filtros
FL1 = J + 1 * corrl1
FL2 = J - 1 * corrl2
FL3 = J + 1 * corrl3

# Desplegar varias imágenes en una sola ventana
fig, axs = plt.subplots(4, 2, figsize=(10, 15))

# Imagen original
axs[0, 0].imshow(imagen_org)
axs[0, 0].set_title('Imagen original')
axs[0, 0].axis('off')

# Imagen original
axs[0, 1].imshow(imagen_org_rgb)
axs[0, 1].set_title('Imagen original-rgb')
axs[0, 1].axis('off')

# Laplaciano 1
axs[1, 0].imshow(corrl1, cmap='gray')
axs[1, 0].set_title('Laplaciano 1')
axs[1, 0].axis('off')

# Filtro L1
axs[1, 1].imshow(FL1, cmap='gray')
axs[1, 1].set_title('Filtro L1')
axs[1, 1].axis('off')

# Laplaciano 2
axs[2, 0].imshow(corrl2, cmap='gray')
axs[2, 0].set_title('Laplaciano 2')
axs[2, 0].axis('off')

# Filtro L2
axs[2, 1].imshow(FL2, cmap='gray')
axs[2, 1].set_title('Filtro L2')
axs[2, 1].axis('off')

# Laplaciano 3
axs[3, 0].imshow(corrl3, cmap='gray')
axs[3, 0].set_title('Laplaciano 3')
axs[3, 0].axis('off')

# Filtro L3
axs[3, 1].imshow(FL3, cmap='gray')
axs[3, 1].set_title('Filtro L3')
axs[3, 1].axis('off')

plt.tight_layout()
plt.show()