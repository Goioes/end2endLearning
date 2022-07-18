from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

def baseModelKeras(networkData=None):
    model = Sequential()
    
    # Normalizing data in range of -1 to 1 and zero-centering data
    model.add(Lambda(lambda x: x / 127.5 - 1.0))

    # Layer 1: 5x5 Conv + ELU + 2x2 MaxPool
    model.add(Conv2D(24, (5,5), strides=(1, 1), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # Layer 2: 5x5 Conv + ELU + 2x2 MaxPool
    model.add(Conv2D(36, (5,5), strides=(1, 1), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # Layer 3: 5x5 Conv + ELU + 2x2 MaxPool
    model.add(Conv2D(48, (5,5), strides=(1, 1), padding='same', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    # Layer 4: 3x3 Conv + ELU + 2x2 MaxPool
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='elu'))

    # Layer 5: 3x3 Conv + ELU + 2x2 MaxPool + Dropout(drop_prob=0.5)
    model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='elu'))
    model.add(Dropout(0.5))

    # Layers 6-8: Fully connected + ELU activation
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    model.add(Dense(50, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(2))

    return model

class BaseModel(Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.cnnLayers = Sequential(
                Conv2d(in_channels=9, out_channels=3, kernel_size=6, stride=1, padding='same'),
                BatchNorm2d(3),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=6, stride=6, padding=0),
                # Another cnn layer
                Conv2d(in_channels=3, out_channels=3, kernel_size=4, stride=1, padding='same'),
                BatchNorm2d(3),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=4, stride=4, padding=0),
                # Another cnn layer
                Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding='same'),
                BatchNorm2d(3),
                ReLU(inplace=True),
                MaxPool2d(kernel_size=5, stride=5, padding=0),
                )
        self.linearLayers = Sequential(
                Linear(3 * 5 * 5, 2)
                )
    def forward(self, x):
        x = self.cnnLayers(x)
        x = x.view(x.size(0), -1)
        x = self.linearLayers(x)
        return x

def baseModel(networkData):
    model = BaseModel()
    if networkData is not None:
        model.load_state_dict(networkData)
    return model

