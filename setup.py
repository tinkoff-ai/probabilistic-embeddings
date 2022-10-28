from setuptools import setup, find_namespace_packages


setup(
    version="0.1.0",
    name="probabilistic_embeddings",
    long_description="Experiments with MDN for metric learning.",
    url="https://github.com/tinkoff-ai/probabilistic-embeddings",
    author="Ivan Karpukhin and Stanislav Dereka",
    author_email="karpuhini@yandex.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "catalyst==21.9",
        "faiss-cpu==1.7.0",  # Need for MAP@R metric computation.
        "jpeg4py==0.1.4",
        "mxnet==1.8.0.post0",  # Used for RecordIO reading.
        "numpy==1.19.5",
        "optuna==2.10.0",
        "pretrainedmodels==0.7.4",
        "scikit-image==0.17.2",
        "scikit-learn==0.24.2",
        "scipy==1.5.4",
        "torch==1.10.1",  # CUDA 10.1.
        "torchvision==0.11.2",
        "Pillow==8.3.1",
        "PyYAML==5.4.1",
        "gitpython",
        "wandb"
    ]
)
