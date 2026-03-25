from setuptools import setup, find_packages

setup(
    name='potential-math',
    version='0.2.0',
    author='王江祁',
    author_email='jq.wang@126.com',
    description='势场数学库 - 基于频率调制统一理论的地形感知优化器',
    long_description=open('README.md', encoding='utf-8').read() if __import__('os').path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://gitee.com/sageapollo/potential-math',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',  # 仅用于示例
    ],
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords='optimizer, deep learning, physics-inspired, potential field, curvature-adaptive',
)