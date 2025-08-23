from google.colab import drive
drive.mount('/content/drive')
folder_path = "/content/drive/MyDrive/CNN_pred/"

from matplotlib import pyplot as plt
import numpy as np
from keras.models import load_model
import scipy.io


nParams=4
res=1280
L=500
Lw=50
Step=4
DLw=int(Lw/Step)
windows_res=int(res*(Lw/L))
Lp= res
ratio = res/L
Lwp= int(ratio * Lw)

dir = folder_path + "InputData/200_Images_L500_LW50_radius_2.5_res_" + str(res) + ".mat"
mat = scipy.io.loadmat(dir)
x_dataALL=mat['Img_array']

for ID in range(0,1):            # NUMBER OF IMAGES TO RUN
    plt.imshow(x_dataALL[0,:,:], interpolation=None) # reconstructs image from pixels, for testing
    plt.show()

    test=x_dataALL[ID,:,:]
    xc=range(int(Lw/2),int(L-(Lw/2)+1),DLw)
    yc=xc
    nw= len(xc)

    i=0

    x_data = np.zeros(((nw*nw,windows_res,windows_res)))
    for x in xc:
        for y in yc:
            xlower=int(round((x-Lw/2)*ratio))
            xupper=xlower + Lwp
            ylower=res-int(round((y-Lw/2)*ratio))
            yupper=ylower - Lwp
            selected_image = test[ yupper:ylower, xlower:xupper]
            x_data[i,:,:] = selected_image
            #plt.imshow(selected_image, interpolation=None) # reconstructs image from pixels, for testing
            #plt.show()
            i=i+1
    x_data = x_data.astype("float32") / 255
    x_data= np.expand_dims(x_data, -1)
    #plt.imshow(x_data[0,:,:], interpolation=None) # reconstructs image from pixels, for testing
    #plt.show()

    print(x_data.shape)
    model = load_model(folder_path + 'OutputData/SavedModels/CNN_model.keras')
    y_norm=model.predict(x_data) # need proper dim
    ################################################################################################################################
    MinMax = np.loadtxt(folder_path + 'OutputData/MinMax.txt',dtype="float")

    miny=MinMax[:,0] if nParams>1 else MinMax[0]
    maxy=MinMax[:,1] if nParams>1 else MinMax[1]

    y_data=np.zeros((nw*nw,nParams))
    for i in range(0,nParams):
        y_data[:,i] = (y_norm)*(maxy - miny)  + miny if nParams==1 else  (y_norm[:,i])*(maxy[i] - miny[i])  + miny[i]

    string1= str(ID+1) + '_predicted_Cij' + '_L_' + str(L) +'_Lw_' + str(Lw) +"_Step_" + str(Step)
    np.savetxt(folder_path + 'OutputData/PredictedCij/' +string1 + '.txt', y_data, delimiter=' ',fmt='%f')   # X is an array
    print(ID)




### COMPARISON TRUE VS PREDICTED CIJ ##
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

## PREDICTED CIJ ##

Xc, Yc = np.meshgrid(xc, yc)

iparam = 0 # C11

C11_pred = np.reshape(y_data[:,iparam], (nw, nw))

# Plot
ax1.plot_surface(Xc, Yc, C11_pred, cmap='viridis')
ax1.set_title('Predicted C11')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')


## TRUE CIJ ##


data = np.loadtxt(folder_path + "OutputData/TrueCij/1SVE_data_Lw=50_Stiffness_ratio_1000_L_500Step_4.txt" )

C11_true = data[:,2] # C11
C11_true = np.reshape(C11_true, (nw, nw))

# Plot
ax2.plot_surface(Xc, Yc, C11_true, cmap='viridis')
ax2.set_title('True C11')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
z_min = min(C11_pred.min(), C11_true.min())
z_max = max(C11_pred.max(), C11_true.max())
ax1.set_zlim(z_min, z_max)
ax2.set_zlim(z_min, z_max)

absolute_percentage_error = np.mean(np.abs((C11_pred - C11_true) / C11_true)) * 100
print(f"Absolute Percentage Error for C11: {absolute_percentage_error:.2f}%")

plt.show()