# CS 4414/5416 Fall 2025 - Assignment 1 Infrastructure

Release Data:  09/03/2025


These instructions assume familiarity with the CS 3410 and their infrastructure (Fall 2024 and beyond); for more details on specific steps or command arguments, please refer to https://www.cs.cornell.edu/courses/cs3410/2024fa/course/infra.html



## Retrieve Starter Code

Download the starter code from the provided GitHub repository onto your local machine:

```
git clone https://github.com/shouxulin/CS4414-HW1
```


## Docker Image Setup
Make sure you have Docker installed and have the Docker desktop application actively running on your machine.* 

To download the Docker container image we have set up:

```
docker pull ghcr.io/shouxulin/cs4414hw1:latest
```

Important Note:
Some students running on macOS may run into malware issues installing Docker via the Docker website: "com.docker.vmnetd" was not opened because it contains malware. This action did not harm your MAC." (please see the first figure in 
[CS 4414_5416 - HW 1 Infrastructre.pdf](./CS%204414_5416%20-%20HW%201%20Infrastructure.pdf))

<!-- ![MACOS Note](images/macos_note.png) -->


Please note that it’s not actually malware – Docker messed up with signing their application in compliance with certain Apple rules. To get around this issue, go to the terminal and run the following commands:

```
brew uninstall --cask docker --force
brew uninstall --formula docker --force
brew install --cask docker
```

The above commands will uninstall the existing Docker installation and install Docker via [homebrew](https://brew.sh). After installation finishes, Docker should pop up in the application folder and you should be able to access the Docker GUI.


## Shortcut to Docker Command (`rv` alias)


We will create an alias to shorten the command that tells Docker to run a given command in the CS 4414 infrastructure container.

### A. Temporary Alias (Current Session Only)
If you want the alias to only last for your current terminal session:

```
alias rv='docker run -i --rm -v "$PWD":/root ghcr.io/shouxulin/cs4414hw1'
```

### B. Persistent Alias (All Future Sessions)
To make it stick around when you open a new terminal window, type this command to find out which shell you’re using:
```
echo $SHELL
```


At the end of the following command, either change to bashrc or zshrc or other according to your shell.
```
echo "alias rv='docker run -i --rm -v \"\$PWD\":/root ghcr.io/shouxulin/cs4414hw1'" >> ~/.zshrc
```


## Example Usage:

### A. Compile & Run
To compile, make sure you are in the part1 directory

```
rv g++ -std=c++20 -O3 -Wall main.cpp knn.cpp -o main
```

NOTE: In Assignment 1, a `Makefile` has been provided for each part to simplify the compilation for you. Instead of running the above, run the following:

```
rv make all
```

To run the newly compiled program, here are example inputs:
```
rv ./main ./data/1d-1.json ./data/1d-100.json 2
```


### B. Debugging (gdb & valgrind)

Please see the last figure in 
[CS 4414_5416 - HW 1 Infrastructre.pdf](./CS%204414_5416%20-%20HW%201%20Infrastructure.pdf)