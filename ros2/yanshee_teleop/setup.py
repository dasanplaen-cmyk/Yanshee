from glob import glob

from setuptools import setup

package_name = "yanshee_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Drive a UBTECH Yanshee with an Xbox controller over ROS 2",
    license="MIT",
    entry_points={"console_scripts": ["driver = yanshee_teleop.driver:main"]},
)
