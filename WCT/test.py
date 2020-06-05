import cv2

img1=cv2.imread("styles/0resize.jpg")
print(img1.shape)
cv2.imshow("",img1)
cv2.waitKey(0)
