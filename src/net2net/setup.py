from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Params Python package'

# Setting up
setup(
       # The name must match the folder name 'net2net'
        name="net2net", 
        version=VERSION,
        author="Tom Ravaud",
        author_email="<tom.ravaud@eleves.enpc.fr>",
        description=DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # Add any additional packages that 
        # needs to be installed along with your package.
        
        keywords=['python', 'net2net'],
        classifiers= [
            "Development Status :: 1 - Planning",
            "License :: OSI Approved :: MIT License",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
        ]
)
