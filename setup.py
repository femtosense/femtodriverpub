"""femtorun setup script"""
import pathlib
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

HERE = pathlib.Path(__file__).parent.resolve()

# Get the package name from the NAME file
NAME = (HERE / 'NAME').read_text(encoding='utf-8').strip()
# Get the package version from the VERSION file
VERSION = (HERE / NAME / 'VERSION').read_text(encoding='utf-8').strip()
# Get the package long description from the README file
LONG_DESCRIPTION = (HERE / 'README.md').read_text(encoding='utf-8')
# Get the Python requirements from the PY_REQUIREMENTS file
PY_REQUIREMENTS = (HERE / NAME / 'PY_REQUIREMENTS').read_text(encoding='utf-8').splitlines()

def run_env_reqs():
    """Run the ENV_REQUIREMENTS.sh script"""
    proc = subprocess.Popen(
        [HERE / NAME / 'ENV_REQUIREMENTS.sh'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    # Can use communicate(input='y\n'.encode()) if the command run requires confirmation
    stdout_data, _ = proc.communicate()
    print('Command output: %s' % stdout_data)
    if proc.returncode != 0:
        raise RuntimeError(
            'ENV_REQUIREMENTS install failed: exit code: %s' % (proc.returncode))

class CustomInstall(install):
    """Set up the environment prior to installation"""
    def run(self):
        run_env_reqs()
        install.run(self)

class CustomDevelop(develop):
    """Set up the environment prior to installation"""
    def run(self):
        run_env_reqs()
        develop.run(self)

setup(
    name=NAME, # don't touch, edit the NAME file to name the project
    version=VERSION, # don't touch, edit the VERSION file to update version number
    description="Femtorun defines the runtime interface for Femtosense software",
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url="https://github.com/femtosense/" + NAME,
    author="Femtosense", # can change to primary maintainer's email
    author_email="info@femtosense.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    cmdclass = {
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=PY_REQUIREMENTS,
    project_urls={
        'Source': 'https://github.com/femtosense/' + NAME,
    }
)
