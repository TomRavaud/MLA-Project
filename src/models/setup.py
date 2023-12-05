from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Models Python package'

# Setting up
setup(
       # The name must match the folder name 'net2net'
        name="models", 
        version=VERSION,
        author="Tom Ravaud",
        author_email="<tom.ravaud@eleves.enpc.fr>",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # Add any additional packages that 
        # needs to be installed along with your package.
        
        keywords=['python', 'models', 'CNN', 'Deep Learning'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "License :: OSI Approved :: MIT License",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
        ]
)
