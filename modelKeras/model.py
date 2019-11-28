import inputData
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

epochs = 50

# tensorboard 
tb = TensorBoard(log_dir='./Graph')
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# load data generators
trainingDataGen, validationDataGen = inputData.getData()
print(trainingDataGen, validationDataGen)

num_classes = 7

model = Sequential()
# layer 1
model.add(Conv2D(64, (3, 3), padding="same", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# layer 2
model.add(Conv2D(128, (3, 3), padding="same", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# layer 3
model.add(Conv2D(512, (3, 3), padding="same", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# layer 4
# model.add(Conv2D(512, (3, 3), padding="same", input_shape=(48, 48, 1)))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())

# Fully connect 1
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
# Fully connect 2
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))
opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

history = model.fit_generator(generator=trainingDataGen,
                                steps_per_epoch=trainingDataGen.n // trainingDataGen.batch_size,
                                epochs=epochs, validation_data=validationDataGen,
                                validation_steps = validationDataGen.n // validationDataGen.batch_size,
                                verbose=1,
                                callbacks=[tb, checkpoint])


model.save('FaceExpressRecon.h5')
