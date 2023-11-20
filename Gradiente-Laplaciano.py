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

# Realzar imagen con el filtro Laplaciano
L3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
Corrl3 = cv2.filter2D(J, -1, L3)
P1FL3 = J + 1 * Corrl3

# Magnitud del gradiente
x = np.array([[-1, 2, -1], [0, 0, 0], [1, 2, 1]])
y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Px = cv2.filter2D(J, -1, x, borderType=cv2.BORDER_REPLICATE)
Py = cv2.filter2D(J, -1, y, borderType=cv2.BORDER_REPLICATE)
Mag = np.sqrt(Px**2 + Py**2)

# Suavizado de magnitud
w = np.ones((3, 3), dtype=float)
Suav = cv2.filter2D(Mag, -1, w, borderType=cv2.BORDER_REPLICATE) * (1/9)

# Mascara = Imagen realzada * Magnitud suavizada
Mask = P1FL3 * Suav

# Máscara + imagen original
GradLap = J + Mask

# Desplegar varias imágenes en una sola ventana
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(imagen_org_rgb)
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(GradLap, cmap='gray')
plt.title('Imagen Gradiente-Laplaciano')
plt.axis('off')

plt.show()
