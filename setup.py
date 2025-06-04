from setuptools import setup, find_packages

setup(
    name="echolip-ai",
    version="0.1.0",
    description="A deep learning model for realistic lip synchronization with audio input",
    author="Krishna-Lasya",
    author_email="krishna@humera.ai",
    url="https://github.com/Krishna-Lasya/EchoLip-AI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "face-alignment>=1.3.0",
        "librosa>=0.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)