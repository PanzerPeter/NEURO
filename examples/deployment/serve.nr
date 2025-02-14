# Model Serving Example
# This example demonstrates how to serve a model with a REST API and monitoring

# 1. Load and Configure Model
model = load_model("models/exported/model.pt");
model.to_inference_mode();

# 2. Server Configuration
server_config = {
    "server": {
        "host": "0.0.0.0",
        "port": 8080,
        "workers": 4,
        "timeout": 30
    },
    "model": {
        "batch_size": 32,
        "max_batch_size": 128,
        "dynamic_batching": true,
        "timeout_ms": 100
    },
    "monitoring": {
        "enabled": true,
        "metrics": [
            "requests_per_second",
            "latency_ms",
            "memory_usage",
            "gpu_utilization",
            "batch_size",
            "queue_length"
        ],
        "prometheus": {
            "enabled": true,
            "port": 9090
        },
        "dashboard": {
            "enabled": true,
            "port": 8050
        }
    },
    "scaling": {
        "enabled": true,
        "min_workers": 1,
        "max_workers": 8,
        "target_latency_ms": 100,
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.2
    },
    "cache": {
        "enabled": true,
        "type": "redis",
        "ttl_seconds": 3600,
        "max_size_mb": 1024
    },
    "security": {
        "authentication": {
            "enabled": true,
            "type": "api_key"
        },
        "rate_limiting": {
            "enabled": true,
            "requests_per_minute": 100
        },
        "cors": {
            "enabled": true,
            "allowed_origins": ["*"]
        }
    }
};

# 3. Custom Middleware
@middleware
def request_validator(request) {
    # Validate input shape
    if request.data.shape != (3, 224, 224) {
        return Error(
            code=400,
            message="Invalid input shape"
        );
    }
    
    # Validate input range
    if not (0 <= request.data <= 1).all() {
        return Error(
            code=400,
            message="Input values must be in range [0, 1]"
        );
    }
    
    return request;
}

@middleware
def response_formatter(response) {
    # Format prediction results
    return {
        "predictions": response.tolist(),
        "confidence": float(response.max()),
        "timestamp": current_timestamp()
    };
}

# 4. Custom Monitoring
@monitor
def batch_size_monitor(metrics) {
    current_batch_size = metrics.current_batch_size;
    if current_batch_size > server_config["model"]["batch_size"] {
        print(f"Warning: Large batch size detected: {current_batch_size}");
    }
}

@monitor
def latency_monitor(metrics) {
    current_latency = metrics.latency_ms;
    if current_latency > server_config["model"]["timeout_ms"] {
        print(f"Warning: High latency detected: {current_latency}ms");
    }
}

# 5. Custom Scaling Logic
@scaling_policy
def adaptive_scaling(metrics) {
    # Calculate load factor
    load_factor = metrics.queue_length / metrics.max_queue_length;
    
    if load_factor > server_config["scaling"]["scale_up_threshold"] {
        return "scale_up";
    } elif load_factor < server_config["scaling"]["scale_down_threshold"] {
        return "scale_down";
    }
    
    return "maintain";
}

# 6. Create Server
server = ModelServer(
    model=model,
    config=server_config
);

# Add middleware
server.add_middleware([
    request_validator,
    response_formatter
]);

# Add monitoring
server.add_monitors([
    batch_size_monitor,
    latency_monitor
]);

# Add scaling policy
server.set_scaling_policy(adaptive_scaling);

# 7. Define Endpoints
@server.endpoint("/predict", methods=["POST"])
def predict(request) {
    # Preprocess input
    input_data = preprocess_input(request.data);
    
    # Run inference
    with server.batch_request() as batch:
        predictions = model(input_data);
    
    # Postprocess output
    result = postprocess_output(predictions);
    
    return result;
}

@server.endpoint("/batch_predict", methods=["POST"])
def batch_predict(request) {
    # Validate batch size
    if len(request.data) > server_config["model"]["max_batch_size"] {
        return Error(
            code=400,
            message="Batch size too large"
        );
    }
    
    # Process batch
    results = [];
    with server.batch_request() as batch:
        for item in request.data {
            input_data = preprocess_input(item);
            predictions = model(input_data);
            results.append(postprocess_output(predictions));
        }
    
    return results;
}

@server.endpoint("/health", methods=["GET"])
def health_check() {
    return {
        "status": "healthy",
        "gpu_memory": get_gpu_memory_usage(),
        "cpu_memory": get_cpu_memory_usage(),
        "uptime": server.get_uptime()
    };
}

# 8. Configure Monitoring Dashboard
dashboard = MonitoringDashboard() {
    # Request metrics
    add_chart("Requests per Second") {
        metric = "requests_per_second";
        chart_type = "line";
        refresh_interval = 1;
    }
    
    # Latency distribution
    add_chart("Latency Distribution") {
        metric = "latency_ms";
        chart_type = "histogram";
        refresh_interval = 5;
    }
    
    # Resource usage
    add_chart("Resource Usage") {
        metrics = ["memory_usage", "gpu_utilization"];
        chart_type = "area";
        refresh_interval = 2;
    }
    
    # Batch size distribution
    add_chart("Batch Size Distribution") {
        metric = "batch_size";
        chart_type = "bar";
        refresh_interval = 5;
    }
    
    # Queue length
    add_chart("Request Queue") {
        metric = "queue_length";
        chart_type = "line";
        refresh_interval = 1;
    }
}

# 9. Start Server
print("Starting model server...");
server.start();

# 10. Generate API Documentation
docs = APIDocumentation() {
    title = "Model Serving API";
    version = "1.0.0";
    
    add_endpoint("/predict") {
        method = "POST";
        description = "Make a single prediction";
        request_schema = {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Input tensor (3x224x224)"
                }
            }
        };
        response_schema = {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "confidence": {"type": "number"},
                "timestamp": {"type": "string"}
            }
        };
    }
    
    add_endpoint("/batch_predict") {
        method = "POST";
        description = "Make batch predictions";
        request_schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"}
                    }
                }
            }
        };
    }
    
    add_endpoint("/health") {
        method = "GET";
        description = "Check server health";
        response_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "gpu_memory": {"type": "number"},
                "cpu_memory": {"type": "number"},
                "uptime": {"type": "number"}
            }
        };
    }
}

docs.save("docs/api.html");

print("Model server running at http://localhost:8080");
print("API documentation available at docs/api.html");
print("Monitoring dashboard at http://localhost:8050"); 