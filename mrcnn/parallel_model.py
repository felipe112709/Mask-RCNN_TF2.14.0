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
        # Call super().__init__ first as required by Keras
        super(ParallelModel, self).__init__()
        
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        
        # Define input layer(s) matching the input layer(s) of the original model
        self.inputs_list = []
        for input_tensor in self.inner_model.inputs:
            input_shape = K.int_shape(input_tensor)
            input_dtype = input_tensor.dtype
            new_input = KL.Input(shape=input_shape[1:], dtype=input_dtype, name=input_tensor.name)
            self.inputs_list.append(new_input)
            
        # Build the parallel model
        self.outputs_list = self.make_parallel()
        
        # Call the Model's build method with the new inputs and outputs
        self._init_graph_network(self.inputs_list, self.outputs_list)
        
    def _redirect_to_inner(self, attrname):
        """Redirect loading and saving methods to the inner model."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)
        
    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        try:
            # Try the standard attribute lookup first
            return super(ParallelModel, self).__getattribute__(attrname)
        except AttributeError:
            if 'load' in attrname or 'save' in attrname:
                return getattr(self.inner_model, attrname)
            raise

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        print("Parallel Model Summary:")
        super(ParallelModel, self).summary(*args, **kwargs)
        print("\nInner Model Summary:")
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Get input_names from the inputs list
        input_names = [input_tensor.name for input_tensor in self.inputs_list]
        
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        with tf.device('/cpu:0'):
            input_slices = {name: tf.split(x, self.gpu_count)
                           for name, x in zip(input_names, self.inputs_list)}

        # Handle single or multiple outputs
        if isinstance(self.inner_model.outputs, list):
            num_outputs = len(self.inner_model.outputs)
            output_names = [output.name for output in self.inner_model.outputs]
        else:
            num_outputs = 1
            output_names = [self.inner_model.output.name]
            
        outputs_all = [[] for _ in range(num_outputs)]

        # Run the model call() on each GPU to place the ops there
        for i in range(self.gpu_count):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('tower_%d' % i):
                    # Run a slice of inputs through this replica
                    inputs_for_replica = []
                    for name, tensor in zip(input_names, self.inputs_list):
                        slice_idx = i
                        slice_input = KL.Lambda(
                            lambda x, idx=slice_idx: input_slices[name][idx],
                            name=f'slice_{name}_{i}'
                        )(tensor)
                        inputs_for_replica.append(slice_input)
                    
                    # Create the model replica and get the outputs
                    outputs = self.inner_model(inputs_for_replica)
                    
                    # Handle case where model has a single output
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                        
                    # Save the outputs for merging back together later
                    for l, o in enumerate(outputs):
                        outputs_all[l].append(o)

        # Merge outputs on CPU
        with tf.device('/cpu:0'):
            merged = []
            for l, outputs in enumerate(outputs_all):
                name = output_names[l] if l < len(output_names) else f'output_{l}'
                
                # Concatenate or average outputs?
                # Outputs usually have a batch dimension and we concatenate
                # across it. If they don't, then the output is likely a loss
                # or a metric value that gets averaged across the batch.
                # Keras expects losses and metrics to be scalars.
                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = KL.Lambda(lambda o: tf.add_n(o) / len(outputs), name=f'mean_{name}')(outputs)
                else:
                    # Concatenate
                    m = KL.Concatenate(axis=0, name=f'concat_{name}')(outputs)
                merged.append(m)
                
        return merged


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
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=50, epochs=10, verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[TensorBoard(log_dir=MODEL_DIR, write_graph=True)]
    )
                       epochs=1,
                       validation_data=(x_test, y_test),
                       callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')])
