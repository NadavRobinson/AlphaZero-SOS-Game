import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
# or:
# print(tf.test.is_gpu_available())
# print(tf.config.experimental.list_physical_devices('GPU'))