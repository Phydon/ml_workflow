import tensorflow as tf


def tf_version():
    print("Version:", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())


def gpu_check():
    print("GPU is", "available" if tf.config.list_physical_devices(
        'GPU') else "NOT AVAILABLE")


if __name__ == "__main__":
    tf_version()
    gpu_check()
