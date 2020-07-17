import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lanelinedetect", # Replace with your own username
    version="0.1.4",
    author="guo.zhiwei",
    author_email="zhiweiguo1991@163.com",
    description="车道线检测",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhiweiguo/kkb_cv/tree/master/practical_course/week_7",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
