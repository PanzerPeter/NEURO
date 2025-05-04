# examples/basics/variables.nr
# Demonstrates variable assignment and printing.

count = 5;
learning_rate = 0.01;
message = "Training parameters:";

print(message);
print("  Count:", count);
print("  LR:", learning_rate);

# Reassignment
count = count * 2;
print("Doubled count:", count); 