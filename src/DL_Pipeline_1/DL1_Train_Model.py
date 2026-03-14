import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Permute, Input, Conv1D, Conv2D, Concatenate, MaxPooling1D, MaxPooling2D, Permute, Reshape, Dense, LayerNormalization, Add, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall, Precision, AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class GPU_CWT_Layer(tf.keras.layers.Layer):
    """
    Custom TensorFlow Layer that performs a Continuous Wavelet Transform (CWT).
    Outputs a 'Channels Last' 2D tensor compatible with Keras Conv2D.
    """
    def __init__(self, frequencies, fs=128.0, w=5.0, **kwargs):
        super(GPU_CWT_Layer, self).__init__(**kwargs)
        self.frequencies = frequencies
        self.fs = fs
        self.w = w
        self.num_freqs = len(frequencies)

    def build(self, input_shape):
        self.channels = input_shape[1]
        self.timesteps = input_shape[2]

        t = np.arange(-self.timesteps // 2, self.timesteps // 2) / self.fs
        real_filters, imag_filters = [], []

        for f in self.frequencies:
            s = self.w / (2 * np.pi * f)
            norm = 1.0 / (np.sqrt(s) * np.pi**0.25)
            
            real_part = norm * np.exp(-0.5 * (t/s)**2) * np.cos(self.w * t / s)
            imag_part = norm * np.exp(-0.5 * (t/s)**2) * np.sin(self.w * t / s)

            real_filters.append(real_part)
            imag_filters.append(imag_part)

        real_filters = np.stack(real_filters, axis=-1)[:, np.newaxis, :]
        imag_filters = np.stack(imag_filters, axis=-1)[:, np.newaxis, :]

        self.real_weights = self.add_weight(name='cwt_real', shape=real_filters.shape,
                                            initializer=tf.constant_initializer(real_filters),
                                            trainable=False, dtype=tf.float32)
        self.imag_weights = self.add_weight(name='cwt_imag', shape=imag_filters.shape,
                                            initializer=tf.constant_initializer(imag_filters),
                                            trainable=False, dtype=tf.float32)
        super(GPU_CWT_Layer, self).build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = tf.reshape(inputs, [-1, self.timesteps, 1])

        real_conv = tf.nn.conv1d(x, self.real_weights, stride=1, padding='SAME')
        imag_conv = tf.nn.conv1d(x, self.imag_weights, stride=1, padding='SAME')

        magnitude = tf.sqrt(tf.square(real_conv) + tf.square(imag_conv) + 1e-8)

        # Output shape here: (Batch, 8 Channels, 256 Timesteps, 20 Frequencies)
        out = tf.reshape(magnitude, [batch_size, self.channels, self.timesteps, self.num_freqs])

        # NEW PERMUTATION: (Batch, 20 Frequencies, 256 Timesteps, 8 Channels)
        # This makes the tensor behave exactly like an image with 8 "color" layers
        out = tf.transpose(out, perm=[0, 3, 2, 1])
        return out
    

# ==========================================
# 1. LOAD AND PREPARE THE DATA
# ==========================================
# 1. LOAD AND PREPARE THE DATA
print("Loading raw Pipeline 2 training tensors...")
data = np.load("dl_training_tensors_pipeline2.npz")
X_train = data['X'] 
y_train = data['y']

print(f"Original Shape: X={X_train.shape}, Y={y_train.shape}")

# Create dataset and apply batching so the neural network can ingest it!
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)


# ==========================================
# 2. BLOCK 4: THE MODALITY BIFURCATION
# ==========================================

# Target 20 frequency bands from 1Hz (Delta) to 40Hz (Gamma)
target_frequencies = np.linspace(1.0, 40.0, num=20)

# The Raw Input Block (Batch, 8 Channels, 256 Timesteps)
input_tensor = Input(shape=(8, 256), name="EEG_Raw_Input")

# --- BRANCH A: TEMPORAL ---
# We use a Keras Permute layer to instantly flip the shape from (8, 256) to (256, 8)
# Now it is perfectly ready for a 1D CNN.
temporal_branch = Permute((2, 1), name="Temporal_Permute")(input_tensor)

# --- BRANCH B: SPECTRAL ---
# Generates the Scalograms instantly on the GPU. 
# Shape is natively (Batch, 20 Frequencies, 256 Timesteps, 8 Channels).
# Now it is perfectly ready for a 2D CNN.
spectral_branch = GPU_CWT_Layer(frequencies=target_frequencies, name="Spectral_CWT")(input_tensor)

print("\nModel Bifurcation Successful!")
print(f"Branch A (Temporal) shape: {temporal_branch.shape}") # Expected: (None, 256, 8)
print(f"Branch B (Spectral) shape: {spectral_branch.shape}") # Expected: (None, 20, 256, 8)



temporal_branch_small = Conv1D(32, kernel_size = 3, activation="relu", padding = "same")(temporal_branch)
temporal_branch_medium = Conv1D(32, kernel_size = 5, activation="relu", padding = "same")(temporal_branch)
temporal_branch_large = Conv1D(32, kernel_size = 7, activation="relu", padding = "same")(temporal_branch)

temporal_inceptron = Concatenate(axis=-1)([temporal_branch_small, temporal_branch_medium, temporal_branch_large])

temporal_pool = MaxPooling1D(pool_size = 2)(temporal_inceptron)


spectral_branch_small = Conv2D(32, kernel_size = (3, 3), activation="relu", padding = "same")(spectral_branch)
spectral_branch_medium = Conv2D(32, kernel_size = (5, 5), activation="relu", padding = "same")(spectral_branch)
spectral_branch_large = Conv2D(32, kernel_size = (7, 7), activation="relu", padding = "same")(spectral_branch)

spectral_inceptron = Concatenate(axis=-1)([spectral_branch_small, spectral_branch_medium, spectral_branch_large])

spectral_pool = MaxPooling2D(pool_size = (2, 2))(spectral_inceptron)

# temporal_pool shape is already (Batch, 128, 96)
# spectral_sequence shape is now (Batch, 128, 960)
spectral_sequence = Reshape((128, 960))(spectral_pool)

TOKEN_SIZE = 256

temporal_tokens = Dense(TOKEN_SIZE, activation='relu')(temporal_pool)
spectral_tokens = Dense(TOKEN_SIZE, activation='relu')(spectral_sequence)

print("\nBlock 6 Dimensionality Encoding Complete!")
print(f"Temporal Tokens shape: {temporal_tokens.shape}") # Expected: (None, 128, 256)
print(f"Spectral Tokens shape: {spectral_tokens.shape}") # Expected: (None, 128, 256)

# Cross Attention Fusion, Q = Temporal (Interegator), K&V = Spectral (Answer Sheet)
NUM_HEADS = 8
fused_tokens = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=TOKEN_SIZE)(query = temporal_tokens, value = spectral_tokens)

print("\nBlock 7 Cross-Attention Fusion Complete!")
print(f"Fused Tokens shape: {fused_tokens.shape}") # Expected: (None, 128, 256)



#2. Add & Norm (The Residual Connection)
fused_added = Add()([temporal_tokens, fused_tokens])
fused_norm = LayerNormalization(epsilon=1e-6)(fused_added)

# 3. Feed-Forward Network (FFN) & Final Norm for Block 7
ffn_out = Dense(TOKEN_SIZE * 2, activation='relu')(fused_norm) # Expand
ffn_out = Dense(TOKEN_SIZE)(ffn_out)                           # Compress back
block_7_final = LayerNormalization(epsilon=1e-6)(Add()([fused_norm, ffn_out]))

# ==========================================
# BLOCK 8: GLOBAL TEMPORAL DECODING
# ==========================================
# 1. Global Multi-Head Self-Attention
# Notice that Query, Key, and Value are ALL the exact same tensor now!
self_attention_out = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=TOKEN_SIZE)(
    query=block_7_final, 
    value=block_7_final
)

# 2. Add & Norm 
block_8_added = Add()([block_7_final, self_attention_out])
block_8_norm = LayerNormalization(epsilon=1e-6)(block_8_added)

# 3. FFN & Final Norm for Block 8
ffn_out_8 = Dense(TOKEN_SIZE * 2, activation='relu')(block_8_norm)
ffn_out_8 = Dense(TOKEN_SIZE)(ffn_out_8)
block_8_final_sequence = LayerNormalization(epsilon=1e-6)(Add()([block_8_norm, ffn_out_8]))

# 4. Dimensionality Collapse (The Global Context Vector)
# Flattens (Batch, 128, 256) into (Batch, 256)
global_context_vector = GlobalAveragePooling1D()(block_8_final_sequence)

print("\nBlock 8 Global Temporal Decoding Complete!")
print(f"Global Context Vector shape: {global_context_vector.shape}") # Expected: (None, 256)

# ==========================================
# BLOCK 9: CLASSIFICATION HEAD & FAR CALCULATION
# ==========================================
# 1. The MLP Hidden Layer (Stepping down from 256 to 64)
mlp_hidden = Dense(64, activation='relu', name="Classifier_Hidden")(global_context_vector)

# 2. Regularization (Crucial for minimizing False Alarms / Alarm Fatigue)
# Drops 30% of connections randomly during training to prevent memorizing noise
mlp_dropout = Dropout(0.3)(mlp_hidden)

# 3. The Final Binary Output (0.0 to 1.0 Probability)
final_output = Dense(1, activation='sigmoid', name="Final_Clinical_Diagnosis")(mlp_dropout)

print("\nBlock 9 Classification Head Complete!")
print(f"Output shape: {final_output.shape}") # Expected: (None, 1)

# ==========================================
# FINAL COMPILATION: THE MASTER MODEL
# ==========================================
# We wrap the entire pipeline from your very first 'input_tensor' to the 'final_output'
spatiotemporal_model = Model(inputs=input_tensor, outputs=final_output, name="Spatiotemporal_EEG_Transformer")

# Compile the model with an optimizer, loss function, and targeted metrics.
# We explicitly track Recall, Precision, and AUC because Accuracy is "deceptive" on imbalanced data.
spatiotemporal_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy', 
        Recall(name='recall'),       # How many real seizures did we catch?
        Precision(name='precision'), # When we sounded the alarm, were we right? (Crucial for FAR)
        AUC(name='auc')              # Overall balance of the model
    ]
)

print("\nModel Successfully Compiled! Ready for Training.")
spatiotemporal_model.summary()

# Training and Stopping Loop

print("\nStarting Training...")
EPOCHS = 150

#Early Stop to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='best_spactiotemporal_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = spatiotemporal_model.fit(
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    class_weight={0: 1.0, 1: 3.0},
    callbacks=[early_stop, checkpoint],
    verbose=1
)

print("\n Training Completed")
