
#import os
#os.environ["KERAS_BACKEND"] = "torch"
#import torch
import keras
import numpy as np
from matplotlib import pyplot as plt
import scipy.io

keras.mixed_precision.set_global_policy("mixed_float16")

################################################ Prepare the data ##################################################

NumImages=4000   # number of data to be used
NumTotalImages= 10000  # number of data available
res=128
icomp=0  # selected stiffness component, icomp = 1:5
nParams=4 # number of selected stiffness components

TestSplit= round(0.8*NumImages) # DATA USED FOR TRAINING

string_list= ['C11', 'C12', 'C13', 'C22', 'C23', 'C33']   # STIFFNESS COMPONENTS 


#x_data = np.zeros(((NumImages,res,res)))
dir = "InputData/" + str(NumTotalImages)  + "_Images_L_50_Lw_50_radius_2.5_res_" + str(res) + ".mat"
mat = scipy.io.loadmat(dir)
x_data=mat['Img_array']
x_data = x_data[0:NumImages,:,:]

plt.imshow(x_data[0,:,:], interpolation=None) # reconstructs image from pixels, for testing
plt.show()


filename = 'InputData/'+ str(NumTotalImages) + '_Params_Stiffness_ratio_100'  + '_Lw_50.txt'
Y = np.loadtxt(filename,dtype="float",max_rows=int(NumImages))

y_data = np.zeros((NumImages,nParams))

if nParams ==1:
    y_data = Y[:,icomp]
elif nParams == 4:
    y_data[:,0] = Y[:,0]
    y_data[:,1] = Y[:,1]
    y_data[:,2] = Y[:,3]
    y_data[:,3] = Y[:,5]
    string_list=['C11','C12','C22','C33']
elif nParams ==6:
    y_data = Y

(y_train, y_test) = (y_data[0:TestSplit], y_data[TestSplit:NumImages]) if nParams <= 1 else (y_data[0:TestSplit,:], y_data[TestSplit:NumImages,:])
(x_train, x_test) = (x_data[0:TestSplit,:,:], x_data[TestSplit:NumImages,:,:])
######################### load volume fraction, only used for plotting     
filename = 'InputData/'+ str(NumTotalImages) + '_VF_Stiffness_ratio_100'  + '_Lw_50.txt'
actual_vf = np.loadtxt(filename,dtype="float",usecols =(0),max_rows=int(NumImages))
[vf_train, vf_test]  =  [actual_vf[0:TestSplit], actual_vf[TestSplit:NumImages]]

############################################ NORMALIZE DATA ##############################################################
minlist = np.min(y_train, axis=0) if nParams > 1 else np.min(y_train)
maxlist = np.max(y_train, axis=0) if nParams > 1 else np.max(y_train)

y_train = (y_train - minlist) / (maxlist - minlist)
y_test = (y_test - minlist) / (maxlist - minlist)

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

############################################# MODEL ARCHITECTURE #######################################

input_shape = (res, res, 1)
filter_size = 2
pooling_size= 2
activation_function = keras.layers.LeakyReLU()

model = keras.Sequential(
    [   keras.Input(shape=input_shape),   
        keras.layers.Conv2D(16, kernel_size=(filter_size, filter_size), activation=activation_function),
        keras.layers.AveragePooling2D(pool_size=(pooling_size, pooling_size)),
        keras.layers.Conv2D(32, kernel_size=(filter_size, filter_size), activation=activation_function),
        keras.layers.AveragePooling2D(pool_size=(pooling_size, pooling_size)),
        keras.layers.Conv2D(64, kernel_size=(filter_size, filter_size), activation=activation_function),
        keras.layers.AveragePooling2D(pool_size=(pooling_size, pooling_size)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=activation_function),
        #layers.Dropout(0.1), 
        keras.layers.Dense(128, activation=activation_function),
        #layers.Dropout(0.1), 
        keras.layers.Dense(64, activation=activation_function),
        #layers.Dropout(0.1), 
        keras.layers.Dense(nParams, activation="linear")  
    ]
)

model.summary()
nepochs=20

######################################### MODEL TRAINING ##################################################################

initial_learning_rate = 0.0001
model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adamax(learning_rate=initial_learning_rate), metrics=[keras.metrics.MeanSquaredError()])

history = model.fit( x_train, y_train, batch_size=32, epochs=nepochs, validation_split=0.2)

model.save('OutputData/SavedModels/CNN_model.keras')

####################################### MODEL EVALUATION #####################################################

score = model.evaluate(x_test, y_test, verbose=1)  # loss is score[0], acc is score[1]
np.savetxt('OutputData/score' + '.txt', score, delimiter=',',fmt='%f')
#np.savetxt('OutputData/SampleIndices' + '.txt', test1, delimiter=',',fmt='%i')
np.savetxt('OutputData/MinMax' + '.txt', np.c_[minlist,maxlist], delimiter=' ',fmt='%f')

y_test_pred = model.predict(x_test) # ON TESTING SET
y_train_pred= model.predict(x_train) # ON TRAINING SET

####################################### ERROR METRIC PLOTS ########################

#history.history['val_loss']
x1=history.history['val_mean_squared_error']
#history.history['loss']
x2= history.history['mean_squared_error']
epochs = range(2,nepochs+1)

fig = plt.figure(figsize=(10,5),dpi=150)
#ax = fig.add_subplot(1,1,1)
plt.plot(epochs,x1[1:], 'o-r', markersize = 4., label='Validation')
plt.legend(loc='best')
plt.plot(epochs,x2[1:], 'o-b', markersize = 4., label='Training')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.title('Error metrics during training')
plt.legend(loc='best')
#plt.text(0, 100, [p1_test,p2_test], fontsize=22, bbox=dict(facecolor='red', alpha=0.5))
plt.savefig("OutputData/Training.png")
np.savetxt('OutputData/history' + '.txt', [x1,x2], delimiter=',',fmt='%f')


########################S######### PREDICTED DATA PLOTS #################################

pred = (y_test_pred * (maxlist - minlist)) + minlist 
actual = (y_test * (maxlist - minlist )) + minlist 
pred_train = (y_train_pred * (maxlist - minlist )) + minlist 
actual_train = (y_train * (maxlist- minlist )) + minlist 

np.savetxt('OutputData/actual.txt', actual, delimiter=',',fmt='%f')
np.savetxt('OutputData/pred.txt', pred, delimiter=',',fmt='%f')

for i in range(0,nParams):
    [min_, max_] = [minlist[i] if nParams>1 else minlist, maxlist[i] if nParams>1 else maxlist] 
        
    fig = plt.figure(figsize=(10,4),dpi=150)
    plt.subplot(1, 2, 1)
    plt.axline((min_, min_), slope=1, linewidth=2, color='r')
    #plt.scatter(y_test,y_test_pred, c=vf_test, ec='k')
    [x, y] = [actual[:,i] if nParams>1 else actual, pred[:,i] if nParams>1 else pred] 
    plt.scatter(x,y, c=vf_test, ec='k')
    plt.ylim([min_, max_])
    plt.xlim([min_, max_])
    plt.axis('square')
    plt.xlabel('Testing set')
    plt.ylabel('Testing set predicted')
    #plt.text(0.25, 0.75, round(mse_test,5), fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

    plt.subplot(1, 2, 2)
    plt.axline((min_, min_), slope=1, linewidth=2, color='r')
    #plt.scatter(y_train,y_train_pred, c=vf_train, ec='k')
    [x, y] = [actual_train[:,i] if nParams>1 else actual_train, pred_train[:,i] if nParams>1 else pred_train] 
    plt.scatter(x,y, c=vf_train, ec='k')
    plt.ylim([min_, max_])
    plt.xlim([min_, max_])
    plt.axis('square')
    plt.xlabel('Training set')
    plt.ylabel('Training set predicted')
    #plt.text(0.25, 0.75, round(mse_train,5), fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

    plt.suptitle( str(string_list[i]), fontsize=14)
    plt.savefig('OutputData/' + str(string_list[i]) + '.jpg')

print('done')
