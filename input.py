test=cv2.imread(r'C:\Users\HP\Downloads\diksha.jpg')
test_img=cv2.resize(test,(256,256))
test_img.resize(1,256,256,3)

labels=['rohit','diksha']

images=[]
for filename in os.listdir(r"D:\dataset"):
    for file in os.listdir(os.path.join(r"D:\dataset",filename)):
        img=cv2.imread(os.path.join(os.path.join(r"D:\dataset",filename),file))
    #print(img.shape)
        if img is not None:
            temp=cv2.resize(img,(256,256))
            images.append(temp)
    


x_train=np.array(images)
#cv2.imshow('k',x_train[0])
y=np.array([[0],[1]])
y_train=np.repeat(y,9)
y_train.resize(18,1)
