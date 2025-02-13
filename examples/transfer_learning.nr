@pretrained("resnet18")
backbone = BACKBONE(freeze=True)

model = SEQUENTIAL(
    backbone,
    FLATTEN(),
    DENSE(in_features=512, out_features=256),
    DROPOUT(p=0.5),
    DENSE(in_features=256, out_features=10)
)

# Configure training
loss = CROSSENTROPY()
optimizer = ADAM(learning_rate=0.001)

# Train for 10 epochs
for epoch in range(10) {
    train(model, loss=loss, optimizer=optimizer)
    evaluate(model)
} 