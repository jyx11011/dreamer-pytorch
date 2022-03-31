import numpy as np
import matplotlib.pyplot as plt
import os

def show(img, hspace=0.05, wspace=0.05):
    if len(img.shape)==4:
        img=np.expand_dims(img, 1)
    elif len(img.shape)==3:
        img=np.expand_dims(img,(0,1))
    row=img.shape[0]
    col=img.shape[1]

    f, axarr = plt.subplots(row,col) 
    f.tight_layout()
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    for i in range(row):
        if col == 1:
            axarr[i].imshow(img[i][0])
            axarr[i].axis('off')
        else:
            for j in range(col):
                if row==1:
                    axarr[j].imshow(img[i][j])
                    axarr[j].axis('off')
                else:
                    axarr[i][j].imshow(img[i][j])
                    axarr[i][j].axis('off')

if __name__=='__main__':
    img=np.transpose(np.load(os.getcwd()+'/dreamer/models/cartpole/cartpole_balance.npy'),
                        (1,2,0))
    show(np.array([img,img]))

    plt.savefig('cartpole.png',bbox_inches='tight')
