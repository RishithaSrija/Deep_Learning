import math

# -----------------------------
# INPUT VALUES (Normalized)
# -----------------------------
x1 = 0.2
x2 = 0.24
x3 = 0.80

target = 1
learning_rate = 0.1

# -----------------------------
# INITIAL WEIGHTS
# Input -> Hidden
# -----------------------------
w_x1_h1 = 0.11
w_x2_h1 = 0.14
w_x3_h1 = 0.17

w_x1_h2 = 0.21
w_x2_h2 = 0.24
w_x3_h2 = 0.27

# Hidden biases
b_h1 = 0.1
b_h2 = 0.1

# -----------------------------
# Hidden -> Output Weights
# -----------------------------
w_h1_o1 = 0.31
w_h2_o1 = 0.34

# Output bias
b_o1 = 0.1

# -----------------------------
# SIGMOID FUNCTION
# -----------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# =====================================================
# TASK 1: FORWARD PROPAGATION
# =====================================================

# Net input to hidden neurons
net_h1 = (x1 * w_x1_h1) + (x2 * w_x2_h1) + (x3 * w_x3_h1) + b_h1
net_h2 = (x1 * w_x1_h2) + (x2 * w_x2_h2) + (x3 * w_x3_h2) + b_h2

print("Net input at h1 =", round(net_h1, 4))
print("Net input at h2 =", round(net_h2, 4))

# Hidden layer outputs
out_h1 = sigmoid(net_h1)
out_h2 = sigmoid(net_h2)

print("\nOutput of h1 =", round(out_h1, 4))
print("Output of h2 =", round(out_h2, 4))

# =====================================================
# TASK 2: OUTPUT LAYER
# =====================================================

# Net input at output neuron
net_o1 = (out_h1 * w_h1_o1) + (out_h2 * w_h2_o1) + b_o1

print("\nNet input at output neuron =", round(net_o1, 4))

# Final predicted output
predicted_output = sigmoid(net_o1)

print("Predicted Output =", round(predicted_output, 4))

# =====================================================
# TASK 3: ERROR CALCULATION
# =====================================================

# Mean Squared Error
mse = 0.5 * ((target - predicted_output) ** 2)

print("\nMean Squared Error =", round(mse, 4))

# =====================================================
# TASK 4: BACKPROPAGATION
# =====================================================

# Output layer gradient (delta)
delta_output = (
    (target - predicted_output)
    * predicted_output
    * (1 - predicted_output)
)

print("\nOutput Layer Delta =", round(delta_output, 6))

# Hidden layer gradients
delta_h1 = (
    out_h1
    * (1 - out_h1)
    * (delta_output * w_h1_o1)
)

delta_h2 = (
    out_h2
    * (1 - out_h2)
    * (delta_output * w_h2_o1)
)

print("Hidden Layer Delta h1 =", round(delta_h1, 6))
print("Hidden Layer Delta h2 =", round(delta_h2, 6))

# =====================================================
# TASK 5: UPDATE WEIGHTS
# =====================================================

# -------- Hidden -> Output weights --------
new_w_h1_o1 = w_h1_o1 + (learning_rate * delta_output * out_h1)
new_w_h2_o1 = w_h2_o1 + (learning_rate * delta_output * out_h2)

# Update output bias
new_b_o1 = b_o1 + (learning_rate * delta_output)

# -------- Input -> Hidden weights --------

# h1 weights
new_w_x1_h1 = w_x1_h1 + (learning_rate * delta_h1 * x1)
new_w_x2_h1 = w_x2_h1 + (learning_rate * delta_h1 * x2)
new_w_x3_h1 = w_x3_h1 + (learning_rate * delta_h1 * x3)

# h2 weights
new_w_x1_h2 = w_x1_h2 + (learning_rate * delta_h2 * x1)
new_w_x2_h2 = w_x2_h2 + (learning_rate * delta_h2 * x2)
new_w_x3_h2 = w_x3_h2 + (learning_rate * delta_h2 * x3)

# Update hidden biases
new_b_h1 = b_h1 + (learning_rate * delta_h1)
new_b_h2 = b_h2 + (learning_rate * delta_h2)

# =====================================================
# DISPLAY UPDATED WEIGHTS
# =====================================================

print("\nUPDATED WEIGHTS")

print("\nHidden -> Output")
print("w_h1_o1 =", round(new_w_h1_o1, 5))
print("w_h2_o1 =", round(new_w_h2_o1, 5))

print("\nInput -> Hidden (h1)")
print("w_x1_h1 =", round(new_w_x1_h1, 5))
print("w_x2_h1 =", round(new_w_x2_h1, 5))
print("w_x3_h1 =", round(new_w_x3_h1, 5))

print("\nInput -> Hidden (h2)")
print("w_x1_h2 =", round(new_w_x1_h2, 5))
print("w_x2_h2 =", round(new_w_x2_h2, 5))
print("w_x3_h2 =", round(new_w_x3_h2, 5))

# =====================================================
# DISPLAY UPDATED BIASES
# =====================================================

print("\nUPDATED BIASES")
print("b_h1 =", round(new_b_h1, 5))
print("b_h2 =", round(new_b_h2, 5))
print("b_o1 =", round(new_b_o1, 5))