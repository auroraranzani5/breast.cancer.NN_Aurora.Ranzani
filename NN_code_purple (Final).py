#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading and preprocessing of dataset
raw_data = load_breast_cancer()
X = raw_data.data
y = raw_data.target

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=55)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training
print("Training the Neural Network...")
mlp = MLPClassifier(hidden_layer_sizes=(16,8),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=55)
mlp.fit(X_train, y_train)
print("Training Complete!")

# Make predictions
predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Final Accuracy: {accuracy * 100:.2f}%")


# Architecture Design
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Text
    ax.text(left, top + 0.5, 'Input Layer\n(30 Features)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='indigo')
    ax.text(left + h_spacing, top + 0.5, 'Hidden 1\n(16 Neurons)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='indigo')
    ax.text(left + h_spacing*2, top + 0.5, 'Hidden 2\n(8 Neurons)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='indigo')
    ax.text(right, top + 0.5, 'Output\n(Diagnosis)', ha='center', va='bottom', fontsize=10, fontweight='bold', color='indigo')

    # Nodes and Edges
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle_x = n * h_spacing + left
            circle_y = layer_top - m * v_spacing
            
            if n > 0: 
                prev_layer_size = layer_sizes[n-1]
                prev_layer_top = v_spacing*(prev_layer_size - 1)/2. + (top + bottom)/2.
                for o in range(prev_layer_size):
                    prev_x = (n-1) * h_spacing + left
                    prev_y = prev_layer_top - o * v_spacing
                    line = plt.Line2D([prev_x, circle_x], [prev_y, circle_y], c='purple', alpha=0.15, lw=0.5)
                    ax.add_artist(line)
            
            # Colours
            if n == 0: circle_color = '#4B0082'   # Indigo
            elif n == 1: circle_color = '#8A2BE2' # BlueViolet
            elif n == 2: circle_color = '#DA70D6' # Orchid 
            else: circle_color = '#800080'        # Purple
            circle = plt.Circle((circle_x, circle_y), v_spacing/4., color=circle_color, ec='indigo', zorder=4)
            ax.add_artist(circle)

# Graph 1 (NN)
fig = plt.figure(figsize=(10, 8))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, [30, 16, 8, 1])
plt.title("Breast Cancer ANN Architecture", fontsize=15, color='indigo')
plt.show()

# Graph 2 (Learning Curve)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, color='purple', linewidth=2)
plt.xlabel('Epochs', fontsize=12, color='purple')
plt.ylabel('Loss Error', fontsize=12, color='purple')
plt.grid(True, linestyle='--', alpha=0.6, color='#d4a5d4') 
plt.savefig('learning_curve_clean.png', dpi=300) 
plt.show()

# Graph 3 (Confusion Matrix)
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, predictions)

# Colours
custom_colors = ["#E0E0E0", "#A0A0B0", "#A0A0B0"]
custom_cmap = mcolors.LinearSegmentedColormap.from_list("GreyishPurpleUniform", custom_colors)

sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.title("Confusion Matrix", color='purple', fontsize=12, fontweight='bold')
plt.xlabel("Predicted", color='purple')
plt.ylabel("Actual", color='purple')
plt.tick_params(colors='purple')
plt.show()
