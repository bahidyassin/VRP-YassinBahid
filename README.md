# VRP Algorithm for Constrained Optimization:

The code presented here presents an original algorithm that hopes to answer the problem presented in problem.txt.
(This project was made in the context of a challenge for Vorto. Thanks to them for providing the problem and data set.)
## Important note:

The code runs fine. However, a line needed to be added to evaluateShare.py. (If using your evaluateShare.py, please add the following in line 80:)


```python
    line = line.replace('\r', '')
  ```
## Running Code with Virtual Env:
In Terminal Run:

```cmd
python -m venv VRPenv
```

Then,

```cmd
.\VRPenv\Scripts\activate
```

Then run:

```cmd
    pip install -r requirements.txt
```

And finally, run:

```cmd
python3 evaluateShared.py --cmd "python VRP.py" --problemDir path of the training problem (expl: './Training Problems')
```

## Pseudo Code Algorithm:
    - Step 1: Find the closest node to the origine (including the distance between pickup and drop off)
    - Step 1: Initialize the drivers' list with one driver with the latest location being the drop-off, the mileage traveled, and routes lists adding a list for each driver added.
    - Step 3: Remove the location reached from the load's left list.
    - While the loads left list is not empty:
        - Step 4.1: Compute all distances between the drivers' current location and all other nodes left.
        - Step 4.2: Sort the results in a list
        - Step 4.3: For strategy in the sorted list (a strategy consisting of the movement of a specific driver from one node to another):
            - Step 4.3.1: If moving to the new location leads to the driver moving more than he is allowed, ignore the strategy.
            - Step 4.3.2: If adding a new driver going to the location is less costly than the strategy, then ignore it.
            - Step 4.3.2: Otherwise, add the location to the driver and update mileage.
            - Step 4.3.3: Remove the latest node from the load's left list.
            - Step 4.3.4: Break the for loop
        - Step 4.4: If no strategy is chosen, Add a new driver going to the closest node to the original (including distance to dropoff)


## File Directory:

$ tree
.
├── Training Problems: Folder with the training problem
│ 
├── problem.txt: problem statement.
│  
├── evaluateShared.py: code to evaluate the algorithm
│
├── requirements.txt: required packages if you want to run the code on the local machine.
│
├── VRP.py : Path-finding Algorithm
│
└── README.md




