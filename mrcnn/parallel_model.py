import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

class ParallelModel(KM.Model):
    """Subclasses the standard Keras Model and adds multi-GPU support.
    It works by creating a copy of the model on each GPU. Then it slices
    the inputs and sends a slice to each copy of the model, and then
    merges the outputs together and applies the loss on the combined
    outputs.
    """

    def __init__(self, keras_model, gpu_count):
        """Class constructor.
        keras_model: The Keras model to parallelize
        gpu_count: Number of GPUs. Must be > 1
        """
        # Initialize parent class
        super(ParallelModel, self).__init__()
        self.inner_model = keras_model
        self.gpu_count = gpu_count

    def call(self, inputs, training=None):
        """Override the call method to implement the multi-GPU functionality."""
        # Convert single input to list for uniform processing
        if not isinstance(inputs, list):
            inputs = [inputs]
            
        # List to collect outputs from each GPU
        all_outputs = []
        
        # Process batch splits on each GPU
        batch_size = tf.shape(inputs[0])[0]
        split_size = batch_size // self.gpu_count
        remainder = batch_size % self.gpu_count
        
        start_idx = 0
        for i in range(self.gpu_count):
            # Calculate correct split size for this GPU
            if i < remainder:
                size = split_size + 1
            else:
                size = split_size
                
            # Skip this GPU if it would get 0 examples
            if size == 0:
                continue
                
            # Calculate end index for this batch
            end_idx = start_idx + size
                
            # Process on specific GPU
            with tf.device(f'/gpu:{i}'):
                # Slice the inputs for this GPU
                gpu_inputs = []
                for inp in inputs:
                    sliced = inp[start_idx:end_idx]
                    gpu_inputs.append(sliced)
                
                # If there's only one input, unpack it from the list
                if len(gpu_inputs) == 1:
                    gpu_inputs = gpu_inputs[0]
                
                # Forward pass through the inner model
                outputs = self.inner_model(gpu_inputs, training=training)
                
                # Convert single output to list for uniform collection
                if not isinstance(outputs, list):
                    outputs = [outputs]
                    
                # First GPU - initialize output lists
                if i == 0:
                    for _ in range(len(outputs)):
                        all_outputs.append([])
                        
                # Collect outputs from this GPU
                for j, output in enumerate(outputs):
                    all_outputs[j].append(output)
            
            # Update start index for next iteration
            start_idx = end_idx
        
        # Combine results from all GPUs
        merged_outputs = []
        
        with tf.device('/cpu:0'):
            for outputs_list in all_outputs:
                # Skip empty lists (could happen if some GPUs got 0 examples)
                if not outputs_list:
                    continue
                    
                # Merge the outputs - use concatenate for normal outputs
                merged = tf.concat(outputs_list, axis=0)
                merged_outputs.append(merged)
            
        # If there's only one output, unpack it from the list
        if len(merged_outputs) == 1:
            return merged_outputs[0]
        else:
            return merged_outputs
            
    def train_step(self, data):
        """Override the train_step method to handle multi-GPU training."""
        # Unpack data if it's a tuple (data, labels)
        if isinstance(data, tuple):
            data = data[0]
            labels = data[1]
        else:
            # For data generators
            labels = None
            
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(data, training=True)
            
            # If labels weren't unpacked earlier, get them now
            if labels is None:
                if isinstance(data, dict):
                    labels = data.get('labels', data.get('y', None))
                
            # Compute loss
            loss = self.compiled_loss(labels, y_pred)
            
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics
        self.compiled_metrics.update_state(labels, y_pred)
        
        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results
    
    def test_step(self, data):
        """Override the test_step method to handle multi-GPU validation."""
        # Unpack data
        if isinstance(data, tuple):
            data = data[0]
            labels = data[1]
        else:
            # For data generators
            labels = None
            
        # Forward pass
        y_pred = self(data, training=False)
        
        # If labels weren't unpacked earlier, get them now
        if labels is None:
            if isinstance(data, dict):
                labels = data.get('labels', data.get('y', None))
            
        # Update metrics
        self.compiled_loss(labels, y_pred)
        self.compiled_metrics.update_state(labels, y_pred)
        
        # Return metrics
        return {m.name: m.result() for m in self.metrics}
    
    def compile(self, **kwargs):
        """Override compile to also compile the inner model."""
        self.inner_model.compile(**kwargs)
        super(ParallelModel, self).compile(**kwargs)
    
    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both models."""
        print("Parallel Model Summary:")
        super(ParallelModel, self).summary(*args, **kwargs)
        print("\nInner Model Summary:")
        self.inner_model.summary(*args, **kwargs)
    
    def save(self, filepath, **kwargs):
        """Override save to save the inner model instead."""
        self.inner_model.save(filepath, **kwargs)
        
    def save_weights(self, filepath, **kwargs):
        """Override save_weights to save inner model weights."""
        self.inner_model.save_weights(filepath, **kwargs)
        
    def load_weights(self, filepath, **kwargs):
        """Override load_weights to load inner model weights."""
        self.inner_model.load_weights(filepath, **kwargs)
        

# Example usage
if __name__ == "__main__":
    import os
    import numpy as np
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import TensorBoard

    GPU_COUNT = 2

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    os.makedirs(MODEL_DIR, exist_ok=True)

    def build_model(x_train, num_classes):
        inputs = KL.Input(shape=x_train.shape[1:], name="input_image")
        x = KL.Conv2D(32, (3, 3), activation='relu', padding="same",
                     name="conv1")(inputs)
        x = KL.Conv2D(64, (3, 3), activation='relu', padding="same",
                     name="conv2")(x)
        x = KL.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
        x = KL.Flatten(name="flat1")(x)
        x = KL.Dense(128, activation='relu', name="dense1")(x)
        x = KL.Dense(num_classes, activation='softmax', name="dense2")(x)

        return KM.Model(inputs, x, name="digit_classifier_model")

    # Load MNIST Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype('float32') / 255
    x_test = np.expand_dims(x_test, -1).astype('float32') / 255

    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    # Build data generator and model
    datagen = ImageDataGenerator()
    base_model = build_model(x_train, 10)
    
    print("Number of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Add multi-GPU support.
    parallel_model = ParallelModel(base_model, GPU_COUNT)

    optimizer = SGD(learning_rate=0.01, momentum=0.9, clipnorm=5.0)

    parallel_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizer, metrics=['accuracy'])

    parallel_model.summary()

    # Train
    parallel_model.fit(
        datagen.flow(x_train, y_train, batch_size=64 * GPU_COUNT),
        steps_per_epoch=50, epochs=10, verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[TensorBoard(log_dir=MODEL_DIR, write_graph=True)]
    )
