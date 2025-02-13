@custom_layer
def ResidualBlock(in_channels, out_channels) {
    conv1 = CONV2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
    norm1 = NORMALIZE(num_features=out_channels)
    conv2 = CONV2D(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
    norm2 = NORMALIZE(num_features=out_channels)
    
    return x -> {
        identity = x
        out = conv1(x)
        out = norm1(out)
        out = conv2(out)
        out = norm2(out)
        return out + identity
    }
}

model = SEQUENTIAL(
    CONV2D(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
    MAXPOOL(kernel_size=3, stride=2, padding=1),
    ResidualBlock(64, 64),
    ResidualBlock(64, 128),
    FLATTEN(),
    DENSE(in_features=128, out_features=10)
) 