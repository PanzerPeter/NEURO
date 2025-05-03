# Custom Layer Examples
# This example demonstrates how to create and use custom layers in NEURO

# 1. Custom Layer Definitions

# Squeeze and Excitation Block
@custom_layer
def SqueezeExcitation(x, reduction_ratio=16) {
    # Store input shape
    channels = x.shape[1];
    
    # Squeeze operation
    squeeze = GlobalAveragePooling2D(keepdims=true)(x);
    
    # Excitation operation
    excitation = Dense(
        units=channels // reduction_ratio,
        activation="relu"
    )(squeeze);
    
    excitation = Dense(
        units=channels,
        activation="sigmoid"
    )(excitation);
    
    # Scale the input
    return x * excitation;
}

# Multi-scale Feature Fusion
@custom_layer
def MultiScaleFusion(features_list) {
    # Adaptive pooling to match sizes
    target_size = features_list[-1].shape[2:];
    
    aligned_features = [];
    for features in features_list {
        if features.shape[2:] != target_size {
            features = AdaptiveAveragePooling2D(
                output_size=target_size
            )(features);
        }
        aligned_features.append(features);
    }
    
    # Concatenate and fuse
    fused = Concatenate(axis=1)(aligned_features);
    fused = Conv2D(
        filters=256,
        kernel_size=1,
        activation="relu"
    )(fused);
    
    return fused;
}

# Self-Attention Block
@custom_layer
def SelfAttention(x, num_heads=8) {
    # Project input to queries, keys, and values
    dim = x.shape[1];
    head_dim = dim // num_heads;
    
    queries = Dense(dim)(x);
    keys = Dense(dim)(x);
    values = Dense(dim)(x);
    
    # Reshape for multi-head attention
    queries = Reshape((num_heads, -1, head_dim))(queries);
    keys = Reshape((num_heads, -1, head_dim))(keys);
    values = Reshape((num_heads, -1, head_dim))(values);
    
    # Compute attention scores
    scores = MatMul()([queries, Transpose(keys, (0, 1, 3, 2))]);
    scores = scores / math.sqrt(head_dim);
    attention = Softmax(axis=-1)(scores);
    
    # Apply attention to values
    output = MatMul()([attention, values]);
    output = Reshape((-1, dim))(output);
    
    return output;
}

# Gated Residual Block
@custom_layer
def GatedResidual(x, filters) {
    # Main branch
    main = Conv2D(filters=filters, kernel_size=3, padding="same")(x);
    main = BatchNorm()(main);
    main = ReLU()(main);
    main = Conv2D(filters=filters, kernel_size=3, padding="same")(main);
    main = BatchNorm()(main);
    
    # Gate branch
    gate = Conv2D(filters=filters, kernel_size=1)(x);
    gate = BatchNorm()(gate);
    gate = Sigmoid()(gate);
    
    # Combine branches
    gated = main * gate;
    
    # Residual connection
    if x.shape[1] != filters {
        x = Conv2D(filters=filters, kernel_size=1)(x);
    }
    
    return x + gated;
}

# Example Usage in a Model
model = NeuralNetwork(input_size=(224, 224, 3), output_size=1000) {
    # Initial convolution
    x = Conv2D(filters=64, kernel_size=7, stride=2, padding="same");
    x = BatchNorm();
    x = ReLU();
    x = MaxPool(pool_size=3, stride=2, padding="same");
    
    # Feature extraction with custom layers
    features = [];
    
    # Stage 1
    x = GatedResidual(filters=64);
    x = SqueezeExcitation(reduction_ratio=16);
    features.append(x);
    
    # Stage 2
    x = Conv2D(filters=128, kernel_size=3, stride=2, padding="same");
    x = GatedResidual(filters=128);
    x = SqueezeExcitation(reduction_ratio=16);
    features.append(x);
    
    # Stage 3
    x = Conv2D(filters=256, kernel_size=3, stride=2, padding="same");
    x = GatedResidual(filters=256);
    x = SelfAttention(num_heads=8);
    features.append(x);
    
    # Multi-scale fusion
    x = MultiScaleFusion(features);
    
    # Classification head
    GlobalAveragePooling2D();
    Dense(units=1000, activation="softmax");
}

# Test the custom layers
test_input = torch.randn(1, 3, 224, 224);
output = model(test_input);

# Analyze layer behavior
model.analyze_layer_outputs([
    "squeeze_excitation_1",
    "self_attention_1",
    "gated_residual_1"
]);

# Visualize attention patterns
model.visualize_attention("self_attention_1", test_input);

# Profile custom layers
model.profile_layers([
    "squeeze_excitation_1",
    "multi_scale_fusion_1",
    "self_attention_1",
    "gated_residual_1"
]);

# Save layer visualizations
plots = MultiPlot(rows=2, cols=2) {
    # Feature maps
    subplot(0, 0) {
        add_feature_maps("gated_residual_1");
        set_title("Gated Residual Features");
    }
    
    # Attention patterns
    subplot(0, 1) {
        add_attention_patterns("self_attention_1");
        set_title("Self-Attention Patterns");
    }
    
    # Channel importance
    subplot(1, 0) {
        add_channel_importance("squeeze_excitation_1");
        set_title("Channel Importance");
    }
    
    # Multi-scale fusion
    subplot(1, 1) {
        add_fusion_visualization("multi_scale_fusion_1");
        set_title("Feature Fusion");
    }
};
plots.save("plots/custom_layers_analysis.png");

# Generate layer documentation
model.generate_layer_docs({
    "squeeze_excitation": {
        "description": true,
        "parameters": true,
        "computation_graph": true,
        "example_usage": true
    },
    "multi_scale_fusion": {
        "description": true,
        "parameters": true,
        "computation_graph": true,
        "example_usage": true
    },
    "self_attention": {
        "description": true,
        "parameters": true,
        "computation_graph": true,
        "example_usage": true
    },
    "gated_residual": {
        "description": true,
        "parameters": true,
        "computation_graph": true,
        "example_usage": true
    },
    "export_format": "pdf"
}); 