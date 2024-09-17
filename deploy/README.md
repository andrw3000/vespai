# Getting VespAI running

## Ansible Playbook

Here are some notes on how you can get the VespAI detection running using the provided/trained VespAI YOLOv5 model, using a simple Ansible playbook which automates some of the setup. This was tested on a Raspberry Pi Model 4B running [Raspberry Pi OS (64 bit)](https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-os-64-bit) (full version, Debian 12 "bookworm", release 2024-07-04).

 1. Download and install Raspberry Pi OS (64 bit) to a microSD card, using [one of the available methods](https://www.raspberrypi.com/documentation/computers/getting-started.html#install-an-operating-system), for example:
    - Using [Raspberry Pi Imager](https://www.raspberrypi.com/documentation/computers/getting-started.html#install-using-imager), which has a simple GUI.
    - Direct download of [Raspberry Pi OS (64 bit)](https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-os-64-bit) and then `dd` on the shell command-line, for example: `$ xzcat ./<image-file>.img.xz | sudo dd of=/dev/<microsd-card-device-file> bs=4M conv=fsync status=progress`

 2. Insert the microSD card into the Raspberry Pi and [step through the initial setup and configuration](https://www.raspberrypi.com/documentation/computers/getting-started.html#set-up-your-raspberry-pi), as appropriate.

 3. Enable the `ssh` server on your Raspberry Pi, using `raspi-config` [interactively](https://www.raspberrypi.com/documentation/computers/configuration.html#ssh) or on the [command line](https://www.raspberrypi.com/documentation/computers/configuration.html#ssh-nonint).

 4. [Install Ansible](https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html) on your control node (which could be a desktop/server machine or another Raspberry Pi). If your control node is running Debian Linux and has the `python-dev` package installed already, this may be as simple as running `$ pip install ansible`.

> [!TIP]
> It is also recommended to configure your Ansible control node and Raspberry Pi to use [public key authentication](https://help.ubuntu.com/community/SSH/OpenSSH/Keys), by generating RSA keys on the control node and transferring/copying the public key to the Raspberry Pi. _This is the recommended method for Ansible authentication, and makes running its playbooks easier (without depending on `sshpass` and interactive password authentication)._

 5. Run the Ansible playbook with:
 ```
 ansible-playbook deploy/rpi-playbook.yml --inventory=<rpi-host>, --extra-vars "user=<username> vespai_repo=<git-repo-url> vespai_branch=<git-repo-branch>"
 ```
 
    Where:
    - `<rpi-host>` is the hostname or IP address of your Raspberry Pi.
    - `<username>` is the name of the primary user you setup on your Raspberry Pi.
    - `<git-repo-url>` is the URL of the git repo, e.g. `https://github.com/andrw3000/vespai.git`
    - `<git-repo-branch` is the branch-name (or other git ref) in the git repo, e.g. `ansible-deployment`

  6. All being well, you should now be able to run the VespAI example detection script, by activating the `venv` that was created by the Ansible playbook and then running the script, for example:
     - `source /opt/vespai-venv/bin/activate`
     - `$ cd /opt/vespai`
     - `$ python monitor/monitor_run.py  --root=/opt/vespai --print --save --save-dir ~/detections`

## Manual setup

Follow steps 1-3 of the above, and then refer to the [`rpi-playbook.yml` Ansible playbook file](rpi-playbook.yml) and the adjacent [`requirements.txt` file](requirements.txt) to manually step through the process of installing git, downloading the VespAI git repo, and using `pip install` (potentially in a virtual environment created with `python -m venv <virtual-environment-name>`) to install requirements.