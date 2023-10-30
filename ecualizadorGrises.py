import cv2

# Cargar una imagen en escala de grises
img_gray = cv2.imread('Spider5.jpg', cv2.IMREAD_GRAYSCALE)

# Realizar la ecualización del histograma
equ_gray = cv2.equalizeHist(img_gray)

# Mostrar la imagen original y la imagen ecualizada
cv2.imshow('Imagen Original', img_gray)
cv2.imshow('Imagen Ecualizada', equ_gray)
cv2.waitKey(0)

# Guardamos la imagen en disco
cv2.imwrite('Spider5Gris.png', equ_gray)

cv2.destroyAllWindows()
