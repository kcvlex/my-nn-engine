# Frontend

- An API with following request/response.
    - Request
        - Model ID
            - The supported models are pre-defined.
                - MNIST
                - ResNet
                - Yolo
                - BERT
                - GPT-2
        - Input Data
            - The input data is sent in the TensorData format.
            - The frontend must parse the provided data into TensorData.
                - For example, if the model is MNIST, the input data should be a 28x28 grayscale image.
                - The frontend converts the image into a tensor of shape (1, 1, 28, 28) with appropriate normalization.
        - Backend
            - CPU or CUDA
    - Response
        - The inference result in TensorData format.
