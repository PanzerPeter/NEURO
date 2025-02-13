@custom_layer
def SpatialPath(in_channels) {
    conv1 = CONV2D(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
    pool = MAXPOOL(kernel_size=3, stride=2, padding=1)
    conv2 = CONV2D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    
    return x -> {
        out = conv1(x)
        out = pool(out)
        return conv2(out)
    }
}

@custom_layer
def ContextPath(in_channels) {
    conv1 = CONV2D(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
    pool1 = MAXPOOL(kernel_size=3, stride=2, padding=1)
    conv2 = CONV2D(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    pool2 = MAXPOOL(kernel_size=3, stride=2, padding=1)
    
    return x -> {
        out = conv1(x)
        out = pool1(out)
        out = conv2(out)
        return pool2(out)
    }
}

model = SEQUENTIAL(
    CONCAT([
        SpatialPath(3),
        ContextPath(3)
    ]),
    CONV2D(in_channels=256, out_channels=128, kernel_size=3, padding=1),
    FLATTEN(),
    DENSE(in_features=128, out_features=10)
) 