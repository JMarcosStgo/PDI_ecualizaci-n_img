import cv2

# Cargar una imagen en color (RGB)
img_rgb = cv2.imread('Spider5.jpg')

# Convertir la imagen a escala de grises
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Realizar la ecualización del histograma en escala de grises
equ_gray = cv2.equalizeHist(img_gray)

# Crear una imagen en RGB con el canal de intensidad ecualizado
equ_rgb = cv2.cvtColor(equ_gray, cv2.COLOR_GRAY2BGR)

# Mostrar la imagen original y la imagen ecualizada
cv2.imshow('Imagen Original', img_rgb)
cv2.imshow('Imagen Ecualizada', equ_rgb)
cv2.waitKey(0)

# Guardamos la imagen en disco
cv2.imwrite('spider5.png', equ_rgb)

cv2.destroyAllWindows()