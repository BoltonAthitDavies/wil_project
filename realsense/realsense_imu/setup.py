import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'realsense_imu'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ambushee',
    maintainer_email='ambushee@todo.todo',
    description='United accel+gyro publisher for Intel RealSense cameras.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_node = realsense_imu.imu_node:main',
            'list_profiles = realsense_imu.list_profiles:main',
        ],
    },
)
