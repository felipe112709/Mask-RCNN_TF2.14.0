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
        self.inner_model = keras_model
        self.gpu_count = gpu_count
        merged_outputs = self.make_parallel()
        super(ParallelModel, self).__init__(inputs=self.inner_model.inputs,
                                            outputs=merged_outputs)

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(ParallelModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper
        and inner models."""
        super(ParallelModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def make_parallel(self):
        """Creates a new wrapper model that consists of multiple replicas of
        the original model placed on different GPUs.
        """
        # Slice inputs. Slice inputs on the CPU to avoid sending a copy
        # of the full inputs to all GPUs. Saves on bandwidth and memory.
        input_slices = {name: tf.split(x, self.gpu_count)
                        for name, x in zip(self.inner_model.input_names,
                                            self.inner_model.inputs)}

        output_names = self.inner_model.output_names
        outputs_all = []
        for i in range(len(self.inner_model.outputs)):
            outputs_all.append([])

        # Iterate over GPUs
        for i in range(self.gpu_count):
            # Get a slice of the inputs
            inputs = {name: input_slices[name][i]
                      for name in self.inner_model.input_names}

            # Call the inner model on this slice of inputs.
            # Results will be on the GPU.
            outputs = self.inner_model(inputs)
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Keep outputs in a list for later merging
            for o in range(len(outputs)):
                outputs_all[o].append(outputs[o])

        # Merge outputs on the CPU
        merged_outputs = [KL.concatenate(outputs_all[i], axis=0)
                          for i in range(len(self.inner_model.outputs))]
        return merged_outputs

# Example Usage (adapted for TensorFlow 2.x)
if __name__ == '__main__':
    # Define a simple Keras model
    def build_model():
        input_tensor = KL.Input(shape=(28, 28, 1))
        x = KL.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = KL.MaxPooling2D((2, 2))(x)
        x = KL.Flatten()(x)
        x = KL.Dense(10, activation='softmax')(x)
        return KM.Model(inputs=input_tensor, outputs=x)

    # Instantiate the base model
    base_model = build_model()

    # Check the number of available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    print("Number of GPUs Available: ", num_gpus)

    if num_gpus > 1:
        # Create a parallel model
        parallel_model = ParallelModel(base_model, num_gpus)

        # Compile the parallel model
        parallel_model.compile(optimizer=tf.keras.optimizers.Adam(),
                                loss='categorical_crossentropy',
                                metrics=['accuracy'])

        # Generate some dummy data
        import numpy as np
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1).astype('float32') / 255.0
        x_test = np.expand_dims(x_test, -1).astype('float32') / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        # Train the parallel model
        batch_size = 64 * num_gpus
        epochs = 1
        parallel_model.fit(x_train, y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(x_test, y_test),
                           callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')])
    else:
        print("Not enough GPUs available for parallel training.")
        # Train the base model on a single GPU
        base_model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        base_model.fit(x_train, y_train,
                       batch_size=64,
                       epochs=1,
                       validation_data=(x_test, y_test),
                       callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./logs')])
