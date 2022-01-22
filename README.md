# Virtual Racing League

This is a collection of Jupyter Notebooks I use to train models for DIY Robocar [Virtual Race League](https://docs.donkeycar.com/guide/virtual_race_league/) events. They are adapted from my GeneralAssembly Data Science Immersive [capstone project](https://github.com/GrantMoe/DSI-Capstone-Project).

My computer runs Ubuntu 20.04. You may have to change the directory commands to suit your operating system.

This _should_ work with any Python version 3.6 or higher (due to the f-strings), but I haven't done any testing.

## Input Data
Input data configuration is an artifact of the way [my client](https://github.com/GrantMoe/donkeysim-client) records things. 00_Data_Processing.ipynb expects a .CSV file with the following columns:

* steering_angle
* throttle
* speed
* **image***
* hit
* time
* accel_x
* accel_y
* accel_z
* gyro_x
* gyro_y
* gyro_z
* gyro_w
* pitch
* yaw
* roll
* cte
* activeNode
* totalNodes
* pos_x
* pos_y
* pos_z
* vel_x
* vel_y
* vel_z
* **lap***
* **folder***

With the exception of **image**, **lap**, and **folder**, these are simply the data contained in the [telemetry messages](https://docs.donkeycar.com/guide/simulator/#api) sent by the [Donkey Simulator](https://docs.donkeycar.com/guide/simulator/).

\* **image**: This is the _filename_ of the image corresponding to each record. This is in place of the base-64 encoded string sent by the simulator. The images themselves are stored in another folder.

\* **lap**: This is the lap from which the record was taken. The sim does not provide lap number in its telemetry messages; my client tracks them separately.

\* **folder**: The folder in which the CSV file containing the record is located.

## Trajectory Plotting Data

To plot trajectory, you need a dataframe with at least **pos_x** and **pos_z**. Beyond that, you should include anything you want to use as a "hue" to for any colormapped plots. By default, I use **steering**, **throttle**, **speed**, and **activeNode**.


* Please note that I have renamed the "steering_angle" column as "steering", as is only right and good.